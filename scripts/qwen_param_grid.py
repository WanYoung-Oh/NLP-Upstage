"""
Qwen 생성 파라미터 그리드 비교 (단계 4-A)

dev.csv에서 단일 프롬프트(base)로 파라미터 조합을 비교합니다.
빠른 비교 후 최적 설정을 8-prompt MBR에 적용할 수 있습니다.

실행:
    cd /data/ephemeral/home/NLP/LLM
    python ../scripts/qwen_param_grid.py \
        --model_path mbr_ensemble/r4b_response_only_ckpt \
        --dev_file ../data/dev.csv \
        --output_csv ../prediction/qwen_param_grid.csv

옵션: 8-prompt MBR 비교 추가
    python ../scripts/qwen_param_grid.py \
        --model_path mbr_ensemble/r4b_response_only_ckpt \
        --dev_file ../data/dev.csv \
        --output_csv ../prediction/qwen_param_grid.csv \
        --run_mbr
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

# ── 경로 설정 ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent        # scripts/
_ROOT = _SCRIPT_DIR.parent                           # NLP/
_LLM_DIR = _ROOT / "LLM"

if str(_LLM_DIR) not in sys.path:
    sys.path.insert(0, str(_LLM_DIR))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_model(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    resolved = Path(model_path).expanduser()
    if not resolved.is_absolute():
        # LLM/ 아래 상대 경로 지원
        cwd_cand = (Path.cwd() / resolved).resolve()
        root_cand = (_LLM_DIR / resolved).resolve()
        resolved = cwd_cand if cwd_cand.exists() else root_cand

    tokenizer = AutoTokenizer.from_pretrained(str(resolved))

    adapter_cfg_path = resolved / "adapter_config.json"
    if adapter_cfg_path.exists():
        with open(adapter_cfg_path) as f:
            cfg = json.load(f)
        base_name = cfg["base_model_name_or_path"]
        print(f"[모델] LoRA adapter: {resolved}")
        print(f"[모델] 베이스: {base_name}")
        device_map = {"": 0} if torch.cuda.is_available() else "auto"
        base = AutoModelForCausalLM.from_pretrained(
            base_name, device_map=device_map, torch_dtype="auto", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, str(resolved))
    else:
        print(f"[모델] Full model: {resolved}")
        device_map = {"": 0} if torch.cuda.is_available() else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            str(resolved), device_map=device_map, torch_dtype="auto", trust_remote_code=True
        )

    model.eval()
    return model, tokenizer


def _generate_one(model, tokenizer, dialogue: str, variant_name: str,
                  max_new_tokens: int, do_sample: bool,
                  temperature: float = 1.0, top_p: float = 1.0) -> str:
    from prompts.mbr_prompts import create_messages
    from prompts.postprocess import postprocess_summary

    messages = create_messages(variant_name, dialogue, topic="")
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen_kwargs: dict = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    generated = out[0][len(inputs.input_ids[0]):]
    summary = tokenizer.decode(generated, skip_special_tokens=True)
    return postprocess_summary(summary)


def _run_config(model, tokenizer, dialogues: list[str], label: str,
                variant: str, max_new_tokens: int, do_sample: bool,
                temperature: float = 1.0, top_p: float = 1.0) -> list[str]:
    preds = []
    for d in tqdm(dialogues, desc=label):
        preds.append(_generate_one(
            model, tokenizer, d, variant, max_new_tokens, do_sample, temperature, top_p
        ))
    return preds


def _run_mbr_config(model, tokenizer, dialogues: list[str], label: str,
                    max_new_tokens: int, do_sample: bool,
                    temperature: float = 1.0, top_p: float = 1.0) -> list[str]:
    """8-prompt MBR 실행."""
    from prompts.mbr_decoding import apply_mbr_to_dataset

    all_preds: dict[str, list[str]] = {}
    variant_names = ["base", "abstract", "oneshot", "narrative", "qa_style",
                     "threeshot", "base_copy", "topic"]
    for v in variant_names:
        preds = _run_config(
            model, tokenizer, dialogues, f"{label} [{v}]",
            v, max_new_tokens, do_sample, temperature, top_p
        )
        all_preds[v] = preds

    import pandas as _pd
    dummy_df = _pd.DataFrame({"dialogue": dialogues})
    final = apply_mbr_to_dataset(dummy_df, all_preds, use_mecab=True, verbose=False)
    return final


def _rouge(preds: list[str], refs: list[str]) -> dict:
    from src.utils.metrics import _rouge_baseline
    preds_s = [p if p.strip() else "." for p in preds]
    refs_s  = [r if r.strip() else "." for r in refs]
    s = _rouge_baseline(preds_s, refs_s)
    return {
        "r1": s["rouge-1"], "r2": s["rouge-2"], "rl": s["rouge-l"],
        "combined": s["rouge-1"] + s["rouge-2"] + s["rouge-l"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dev_file", default="../data/dev.csv")
    parser.add_argument("--output_csv", default="../prediction/qwen_param_grid.csv")
    parser.add_argument("--run_mbr", action="store_true",
                        help="8-prompt MBR도 비교 (시간 8× 증가)")
    parser.add_argument("--n_samples", type=int, default=0,
                        help="빠른 테스트: dev 샘플 수 제한 (0=전체)")
    args = parser.parse_args()

    # ── 데이터 로드 ────────────────────────────────────────────────────────
    dev_path = Path(args.dev_file)
    if not dev_path.is_absolute():
        dev_path = (Path.cwd() / dev_path).resolve()
        if not dev_path.exists():
            dev_path = (_ROOT / "data" / "dev.csv").resolve()

    dev_df = pd.read_csv(dev_path)
    if args.n_samples > 0:
        dev_df = dev_df.head(args.n_samples)
    dialogues = dev_df["dialogue"].tolist()
    refs      = dev_df["summary"].tolist()
    print(f"[Dev] {len(dev_df)}건")

    # ── 모델 로드 ──────────────────────────────────────────────────────────
    model, tokenizer = _load_model(args.model_path)

    # ── 비교 설정 목록 ─────────────────────────────────────────────────────
    # (label, variant, max_new_tokens, do_sample, temperature, top_p)
    configs = [
        ("greedy  max128 base", "base", 128, False, 1.0, 1.0),
        ("greedy  max256 base", "base", 256, False, 1.0, 1.0),
        ("t=0.7 p=0.95 max256 base", "base", 256, True, 0.7, 0.95),
        ("t=0.6 p=0.90 max256 base", "base", 256, True, 0.6, 0.90),
    ]

    # ── 추론 & 평가 ────────────────────────────────────────────────────────
    results = []
    all_preds: dict[str, list[str]] = {}

    print("\n" + "=" * 70)
    for label, variant, mnt, ds, temp, tp in configs:
        preds = _run_config(model, tokenizer, dialogues, label,
                            variant, mnt, ds, temp, tp)
        scores = _rouge(preds, refs)
        scores["label"] = label
        results.append(scores)
        all_preds[label] = preds
        print(f"[{label}]  R1={scores['r1']:.4f}  R2={scores['r2']:.4f}"
              f"  RL={scores['rl']:.4f}  Combined={scores['combined']:.4f}")

    # ── 8-prompt MBR (옵션) ────────────────────────────────────────────────
    if args.run_mbr:
        for mnt, ds, temp, tp in [(128, False, 1.0, 1.0), (256, False, 1.0, 1.0)]:
            label = f"8-MBR greedy max{mnt}"
            preds = _run_mbr_config(model, tokenizer, dialogues, label,
                                    mnt, ds, temp, tp)
            scores = _rouge(preds, refs)
            scores["label"] = label
            results.append(scores)
            all_preds[label] = preds
            print(f"[{label}]  R1={scores['r1']:.4f}  R2={scores['r2']:.4f}"
                  f"  RL={scores['rl']:.4f}  Combined={scores['combined']:.4f}")

    # ── 결과 저장 ──────────────────────────────────────────────────────────
    out_path = Path(args.output_csv)
    if not out_path.is_absolute():
        out_path = (_ROOT / "prediction" / out_path.name).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame(results)[["label", "r1", "r2", "rl", "combined"]]
    result_df.to_csv(out_path, index=False)
    print(f"\n결과 저장: {out_path}")

    # 최적 설정 출력
    best = result_df.loc[result_df["combined"].idxmax()]
    print(f"\n▶ 최적 설정: [{best['label']}]  Combined={best['combined']:.4f}")

    # 예측 결과 저장 (최적 설정)
    best_preds = all_preds[best["label"]]
    pred_out = out_path.parent / "qwen_param_best.csv"
    pred_df = dev_df[["fname"]].copy()
    pred_df["summary"] = best_preds
    pred_df.to_csv(pred_out, index=False)
    print(f"최적 예측 저장: {pred_out}")


if __name__ == "__main__":
    main()
