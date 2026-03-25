"""
Qwen dev 추론 (MBR top-5) + KoBART × Qwen 최적 가중치 확정

1. Qwen MBR top-5 프롬프트로 dev 499샘플 추론 → prediction/qwen_dev_mbr.csv
2. KoBART single / kfold dev Combined 계산
3. 가중치 그리드 (0.1~0.5 step 0.1) × dev ROUGE → 최적 가중치 출력

실행:
    cd /data/ephemeral/home/NLP
    python scripts/qwen_dev_ensemble_weight.py \
        --model_path LLM/mbr_ensemble/r4b_response_only_ckpt \
        --kobart_single_ckpt checkpoints/260324_run_003/epoch06_0.7962 \
        --kobart_kfold_csv   prediction/kobart_kfold_test.csv
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
_LLM_DIR = _ROOT / "LLM"
for p in [str(_LLM_DIR), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

MBR_TOP5 = ["qa_style", "base", "base_copy", "topic", "narrative"]


# ── 공통 유틸 ────────────────────────────────────────────────────────────────

def _rouge(preds, refs):
    from src.utils.metrics import _rouge_baseline
    s = _rouge_baseline(
        [p if p.strip() else "." for p in preds],
        [r if r.strip() else "." for r in refs],
    )
    return {"r1": s["rouge-1"], "r2": s["rouge-2"], "rl": s["rouge-l"],
            "combined": s["rouge-1"] + s["rouge-2"] + s["rouge-l"]}


# ── Qwen 모델 로드 ───────────────────────────────────────────────────────────

def _load_qwen(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    resolved = Path(model_path)
    if not resolved.is_absolute():
        resolved = _ROOT / resolved
    resolved = resolved.resolve()

    tokenizer = AutoTokenizer.from_pretrained(str(resolved))

    if (resolved / "adapter_config.json").exists():
        with open(resolved / "adapter_config.json") as f:
            cfg = json.load(f)
        base_name = cfg["base_model_name_or_path"]
        print(f"[Qwen] LoRA: {resolved.name}  base: {base_name}")
        device_map = {"": 0} if torch.cuda.is_available() else "auto"
        base = AutoModelForCausalLM.from_pretrained(
            base_name, device_map=device_map, torch_dtype="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, str(resolved))
    else:
        device_map = {"": 0} if torch.cuda.is_available() else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            str(resolved), device_map=device_map, torch_dtype="auto", trust_remote_code=True
        )

    model.eval()
    return model, tokenizer


def _generate_one(model, tokenizer, dialogue: str, variant: str,
                  max_new_tokens: int = 128) -> str:
    from prompts.mbr_prompts import create_messages
    from prompts.postprocess import postprocess_summary

    messages = create_messages(variant, dialogue, topic="")
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][len(inputs.input_ids[0]):]
    return postprocess_summary(tokenizer.decode(generated, skip_special_tokens=True))


# ── KoBART dev 추론 ──────────────────────────────────────────────────────────

def _kobart_dev_preds(ckt_path: str, dev_df: pd.DataFrame) -> list[str]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from src.data.preprocess import apply_tta
    from src.ensemble import MBRDecoder
    from src.utils.device import get_device
    from src.utils.postprocess import batch_postprocess

    special_tokens = [
        "#Person1#", "#Person2#", "#Person3#", "#Person4#",
        "#Person5#", "#Person6#", "#Person7#",
        "#PhoneNumber#", "#Address#", "#PassportNumber#",
        "#DateOfBirth#", "#SSN#", "#CardNumber#", "#CarNumber#", "#Email#",
    ]
    resolved = ckt_path if Path(ckt_path).is_absolute() else str(_ROOT / ckt_path)
    tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model = AutoModelForSeq2SeqLM.from_pretrained(resolved)
    model.resize_token_embeddings(len(tokenizer))
    device = get_device()
    model.to(device).eval()

    dialogues = dev_df["dialogue"].tolist()
    tta_variants = apply_tta(dialogues, n_ways=2)
    mbr = MBRDecoder()
    remove_tokens = ["<s>", "</s>", "<pad>", "<unk>"]
    preds = []

    for variants in tqdm(tta_variants, desc="KoBART dev TTA"):
        cands = []
        for v in variants:
            tok = tokenizer(
                [v], return_tensors="pt", padding=True,
                truncation=True, max_length=512,
                add_special_tokens=True, return_token_type_ids=False,
            )
            with torch.no_grad():
                gen = model.generate(
                    input_ids=tok["input_ids"].to(device),
                    attention_mask=tok["attention_mask"].to(device),
                    num_beams=4, max_new_tokens=100,
                    no_repeat_ngram_size=3, early_stopping=True,
                )
            cands.append(tokenizer.decode(gen[0], skip_special_tokens=True))
        preds.append(mbr.decode(cands))

    return batch_postprocess(preds, remove_tokens)


# ── 앙상블 가중치 그리드 ─────────────────────────────────────────────────────

def _weighted_ensemble_preds(preds_a: list[str], preds_b: list[str],
                              w_a: float, refs: list[str]) -> dict:
    """토큰 비율 기반 소프트 앙상블 (ROUGE 최적화용)."""
    from src.ensemble import WeightedEnsemble

    df_a = pd.DataFrame({"fname": range(len(preds_a)), "summary": preds_a})
    df_b = pd.DataFrame({"fname": range(len(preds_b)), "summary": preds_b})
    ensemble = WeightedEnsemble()
    merged = ensemble.predict([df_a, df_b], weights=[w_a, round(1.0 - w_a, 1)])
    scores = _rouge(merged["summary"].tolist(), refs)
    return scores


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--kobart_single_ckpt",
                        default="checkpoints/260324_run_003/epoch06_0.7962")
    parser.add_argument("--kobart_kfold_csv",
                        default="prediction/kobart_kfold_test.csv")
    parser.add_argument("--dev_csv", default="data/dev.csv")
    parser.add_argument("--output_dir", default="prediction")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    out_dir = _ROOT / args.output_dir
    dev_df = pd.read_csv(_ROOT / args.dev_csv)
    refs = dev_df["summary"].tolist()
    print(f"[Dev] {len(dev_df)}건")

    # ── 1. Qwen dev MBR top-5 추론 ──────────────────────────────────────────
    qwen_dev_path = out_dir / "qwen_dev_mbr.csv"
    if qwen_dev_path.exists():
        print(f"\n[Qwen dev] 캐시 로드: {qwen_dev_path}")
        qwen_preds = pd.read_csv(qwen_dev_path)["summary"].tolist()
    else:
        print("\n[Qwen dev] MBR top-5 추론 시작...")
        model, tokenizer = _load_qwen(args.model_path)

        all_preds: dict[str, list[str]] = {}
        for variant in MBR_TOP5:
            preds = []
            for d in tqdm(dev_df["dialogue"].tolist(), desc=f"Qwen/{variant}"):
                preds.append(_generate_one(model, tokenizer, d, variant,
                                           args.max_new_tokens))
            all_preds[variant] = preds

        from prompts.mbr_decoding import apply_mbr_to_dataset
        qwen_preds = apply_mbr_to_dataset(dev_df, all_preds, use_mecab=True, verbose=False)

        qwen_dev_df = dev_df[["fname"]].copy()
        qwen_dev_df["summary"] = qwen_preds
        qwen_dev_df.to_csv(qwen_dev_path, index=False)
        print(f"  저장: {qwen_dev_path}")

        # GPU 메모리 해제 (KoBART 추론 전)
        del model
        torch.cuda.empty_cache()

    qw_scores = _rouge(qwen_preds, refs)
    print(f"\n[Qwen dev MBR top-5]  Combined={qw_scores['combined']:.4f}  "
          f"R1={qw_scores['r1']:.4f}  R2={qw_scores['r2']:.4f}  RL={qw_scores['rl']:.4f}")

    # ── 2. KoBART single dev 추론 ────────────────────────────────────────────
    kobart_dev_path = out_dir / "kobart_single_dev.csv"
    if kobart_dev_path.exists():
        print(f"\n[KoBART dev] 캐시 로드: {kobart_dev_path}")
        kobart_preds = pd.read_csv(kobart_dev_path)["summary"].tolist()
    else:
        print("\n[KoBART dev] TTA 추론 시작...")
        kobart_preds = _kobart_dev_preds(args.kobart_single_ckpt, dev_df)
        kb_dev_df = dev_df[["fname"]].copy()
        kb_dev_df["summary"] = kobart_preds
        kb_dev_df.to_csv(kobart_dev_path, index=False)
        print(f"  저장: {kobart_dev_path}")

    kb_scores = _rouge(kobart_preds, refs)
    print(f"[KoBART single dev]   Combined={kb_scores['combined']:.4f}  "
          f"R1={kb_scores['r1']:.4f}  R2={kb_scores['r2']:.4f}  RL={kb_scores['rl']:.4f}")

    # ── 3. KoBART kfold dev 예측 (test CSV를 dev 정렬로 재활용 불가 → single만 사용)
    # kfold test CSV는 test셋 예측이므로 dev ROUGE 계산 불가.
    # 대신 kfold와 single의 test CSV ROUGE는 동일하게 가중치 탐색 결과 적용 가능.

    # ── 4. 가중치 그리드 ─────────────────────────────────────────────────────
    weights = [0.1, 0.2, 0.3, 0.4, 0.5]
    print("\n" + "=" * 65)
    print("▶ 가중치 그리드 (KoBART single + Qwen, dev ROUGE 기준)")
    print(f"{'kobart_w':>10} {'qwen_w':>8} {'R1':>8} {'R2':>8} {'RL':>8} {'Combined':>10}")
    print("=" * 65)

    grid_results = []
    for w_kb in weights:
        w_qw = round(1.0 - w_kb, 1)
        scores = _weighted_ensemble_preds(kobart_preds, qwen_preds, w_kb, refs)
        scores.update({"kobart_w": w_kb, "qwen_w": w_qw})
        grid_results.append(scores)
        print(f"  {w_kb:>8.1f}   {w_qw:>6.1f}   "
              f"{scores['r1']:>6.4f}   {scores['r2']:>6.4f}   "
              f"{scores['rl']:>6.4f}   {scores['combined']:>8.4f}")

    print("=" * 65)

    # ── 최적 가중치 ──────────────────────────────────────────────────────────
    best = max(grid_results, key=lambda x: x["combined"])
    print(f"\n▶ 최적 가중치: kobart={best['kobart_w']}  qwen={best['qwen_w']}")
    print(f"  dev Combined = {best['combined']:.4f}")

    # ── 결과 저장 ────────────────────────────────────────────────────────────
    grid_df = pd.DataFrame(grid_results)[
        ["kobart_w", "qwen_w", "r1", "r2", "rl", "combined"]
    ]
    grid_path = out_dir / "ensemble_weight_grid.csv"
    grid_df.to_csv(grid_path, index=False)
    print(f"\n그리드 저장: {grid_path}")
    print(f"\n→ 최적 앙상블 파일:")
    print(f"    kobart single: prediction/ensemble_kobart{best['kobart_w']}_qwen{best['qwen_w']}.csv")
    print(f"    kobart kfold:  prediction/ensemble_kfold{best['kobart_w']}_qwen{best['qwen_w']}.csv")


if __name__ == "__main__":
    main()
