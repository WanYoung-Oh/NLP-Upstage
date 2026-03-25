"""
Qwen 프롬프트별 ROUGE 비교 (단계 4-B)

8가지 프롬프트를 각각 dev에서 평가하고, 상위 N종으로 MBR 재실행해
노이즈 감소 여부를 확인합니다.

실행:
    cd /data/ephemeral/home/NLP/LLM
    python ../scripts/qwen_prompt_grid.py \
        --model_path mbr_ensemble/r4b_response_only_ckpt \
        --dev_file ../data/dev.csv \
        --output_csv ../prediction/qwen_prompt_grid.csv

옵션: MBR 비교 포함 (상위 k 프롬프트 vs 전체 8개)
    python ../scripts/qwen_prompt_grid.py \
        --model_path mbr_ensemble/r4b_response_only_ckpt \
        --dev_file ../data/dev.csv \
        --output_csv ../prediction/qwen_prompt_grid.csv \
        --run_mbr --top_k 5
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

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
_LLM_DIR = _ROOT / "LLM"

if str(_LLM_DIR) not in sys.path:
    sys.path.insert(0, str(_LLM_DIR))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ALL_PROMPTS = ["base", "abstract", "oneshot", "topic", "narrative",
               "qa_style", "threeshot", "base_copy"]


def _load_model(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    resolved = Path(model_path).expanduser()
    if not resolved.is_absolute():
        for base in [Path.cwd(), _LLM_DIR]:
            cand = (base / resolved).resolve()
            if cand.exists():
                resolved = cand
                break

    tokenizer = AutoTokenizer.from_pretrained(str(resolved))

    if (resolved / "adapter_config.json").exists():
        with open(resolved / "adapter_config.json") as f:
            cfg = json.load(f)
        base_name = cfg["base_model_name_or_path"]
        print(f"[모델] LoRA: {resolved.name}  베이스: {base_name}")
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
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = out[0][len(inputs.input_ids[0]):]
    return postprocess_summary(tokenizer.decode(generated, skip_special_tokens=True))


def _rouge(preds: list[str], refs: list[str]) -> dict:
    from src.utils.metrics import _rouge_baseline
    preds_s = [p if p.strip() else "." for p in preds]
    refs_s  = [r if r.strip() else "." for r in refs]
    s = _rouge_baseline(preds_s, refs_s)
    return {
        "r1": s["rouge-1"], "r2": s["rouge-2"], "rl": s["rouge-l"],
        "combined": s["rouge-1"] + s["rouge-2"] + s["rouge-l"],
    }


def _run_mbr(all_preds: dict[str, list[str]], dialogues: list[str],
             prompt_names: list[str], label: str) -> tuple[list[str], str]:
    from prompts.mbr_decoding import apply_mbr_to_dataset
    import pandas as _pd

    subset = {k: all_preds[k] for k in prompt_names}
    dummy_df = _pd.DataFrame({"dialogue": dialogues})
    final = apply_mbr_to_dataset(dummy_df, subset, use_mecab=True, verbose=False)
    return final, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dev_file", default="../data/dev.csv")
    parser.add_argument("--output_csv", default="../prediction/qwen_prompt_grid.csv")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--run_mbr", action="store_true",
                        help="MBR 비교 실행 (전체 8종 vs 상위 top_k종)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="MBR에 사용할 상위 프롬프트 수")
    parser.add_argument("--n_samples", type=int, default=0,
                        help="빠른 테스트: 0=전체")
    args = parser.parse_args()

    # ── 데이터 로드 ────────────────────────────────────────────────────────
    dev_path = Path(args.dev_file).expanduser()
    if not dev_path.is_absolute():
        for base in [Path.cwd(), _ROOT / "data"]:
            cand = (base / dev_path.name).resolve()
            if cand.exists():
                dev_path = cand
                break

    dev_df = pd.read_csv(dev_path)
    if args.n_samples > 0:
        dev_df = dev_df.head(args.n_samples)
    dialogues = dev_df["dialogue"].tolist()
    refs      = dev_df["summary"].tolist()
    print(f"[Dev] {len(dev_df)}건  max_new_tokens={args.max_new_tokens}")

    # ── 모델 로드 ──────────────────────────────────────────────────────────
    model, tokenizer = _load_model(args.model_path)

    # ── 프롬프트별 추론 ────────────────────────────────────────────────────
    all_preds: dict[str, list[str]] = {}
    single_results = []

    print("\n" + "=" * 70)
    print("▶ 프롬프트별 단독 ROUGE")
    print("=" * 70)

    for prompt_name in ALL_PROMPTS:
        preds = []
        for d in tqdm(dialogues, desc=f"{prompt_name:12s}"):
            preds.append(_generate_one(model, tokenizer, d, prompt_name,
                                       args.max_new_tokens))
        all_preds[prompt_name] = preds

        scores = _rouge(preds, refs)
        scores["prompt"] = prompt_name
        single_results.append(scores)
        print(f"  [{prompt_name:12s}]  R1={scores['r1']:.4f}  "
              f"R2={scores['r2']:.4f}  RL={scores['rl']:.4f}  "
              f"Combined={scores['combined']:.4f}")

    # 단독 성능 순위
    single_df = pd.DataFrame(single_results)[["prompt", "r1", "r2", "rl", "combined"]]
    single_df = single_df.sort_values("combined", ascending=False).reset_index(drop=True)
    print(f"\n▶ 단독 성능 순위 (Combined 기준)")
    for i, row in single_df.iterrows():
        print(f"  {i+1}. {row['prompt']:12s}  {row['combined']:.4f}")

    # ── MBR 비교 (선택) ────────────────────────────────────────────────────
    mbr_results = []
    if args.run_mbr:
        print("\n" + "=" * 70)
        print("▶ MBR 비교")
        print("=" * 70)

        # 전체 8종 MBR
        preds_8, label_8 = _run_mbr(all_preds, dialogues, ALL_PROMPTS, "MBR 전체 8종")
        scores_8 = _rouge(preds_8, refs)
        scores_8["label"] = label_8
        mbr_results.append(scores_8)
        print(f"  [{label_8}]  R1={scores_8['r1']:.4f}  R2={scores_8['r2']:.4f}  "
              f"RL={scores_8['rl']:.4f}  Combined={scores_8['combined']:.4f}")

        # 상위 top_k 종 MBR
        top_prompts = single_df.head(args.top_k)["prompt"].tolist()
        label_k = f"MBR 상위 {args.top_k}종 ({', '.join(top_prompts)})"
        preds_k, _ = _run_mbr(all_preds, dialogues, top_prompts, label_k)
        scores_k = _rouge(preds_k, refs)
        scores_k["label"] = label_k
        mbr_results.append(scores_k)
        print(f"  [{label_k}]\n    R1={scores_k['r1']:.4f}  R2={scores_k['r2']:.4f}  "
              f"RL={scores_k['rl']:.4f}  Combined={scores_k['combined']:.4f}")

        delta = scores_k["combined"] - scores_8["combined"]
        print(f"\n  Δcombined (top{args.top_k} vs 전체8) = {delta:+.4f}"
              f"  ({'축소 MBR 유리' if delta > 0 else '전체 MBR 유지'})")

    # ── 결과 저장 ──────────────────────────────────────────────────────────
    out_path = Path(args.output_csv)
    if not out_path.is_absolute():
        out_path = (_ROOT / "prediction" / out_path.name).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for row in single_results:
        all_rows.append({
            "type": "single", "label": row["prompt"],
            "r1": row["r1"], "r2": row["r2"], "rl": row["rl"], "combined": row["combined"]
        })
    for row in mbr_results:
        all_rows.append({
            "type": "mbr", "label": row["label"],
            "r1": row["r1"], "r2": row["r2"], "rl": row["rl"], "combined": row["combined"]
        })

    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
