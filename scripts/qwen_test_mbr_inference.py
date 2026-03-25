"""
Qwen test셋 MBR top-5 추론

5가지 프롬프트(qa_style, base, base_copy, topic, narrative)로 test.csv 추론 후
MBR 앙상블 → prediction/qwen_test_mbr_top5.csv 저장

실행:
    cd /data/ephemeral/home/NLP
    python scripts/qwen_test_mbr_inference.py \
        --model_path LLM/mbr_ensemble/r4b_response_only_ckpt
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_csv", default="data/test.csv")
    parser.add_argument("--output_csv", default="prediction/qwen_test_mbr_top5.csv")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    test_df = pd.read_csv(_ROOT / args.test_csv)
    print(f"[Test] {len(test_df)}건")

    model, tokenizer = _load_qwen(args.model_path)

    # ── 프롬프트별 추론 ───────────────────────────────────────────────────────
    all_preds: dict[str, list[str]] = {}
    for variant in MBR_TOP5:
        preds = []
        for d in tqdm(test_df["dialogue"].tolist(), desc=f"Qwen/{variant}"):
            preds.append(_generate_one(model, tokenizer, d, variant, args.max_new_tokens))
        all_preds[variant] = preds
        print(f"  ✓ {variant} 완료 ({len(preds)}건)")

    # ── MBR 앙상블 ──────────────────────────────────────────────────────────
    print("\n[MBR top-5 앙상블 중...]")
    from prompts.mbr_decoding import apply_mbr_to_dataset
    final_preds = apply_mbr_to_dataset(test_df, all_preds, use_mecab=True, verbose=True)

    # ── 저장 ─────────────────────────────────────────────────────────────────
    out_path = _ROOT / args.output_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df = test_df[["fname"]].copy()
    result_df["summary"] = final_preds
    result_df.to_csv(out_path, index=False)
    print(f"\n저장: {out_path}  ({len(result_df)}건)")


if __name__ == "__main__":
    main()
