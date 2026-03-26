"""
Unsloth 없는 독립 추론 워커 — subprocess로 호출됩니다.
AutoModelForCausalLM + BitsAndBytesConfig + PeftModel 사용.

사용법 (직접 호출 금지, run_test_inference.py가 subprocess로 실행):
    python inference_worker.py \
        --lora_path <path> \
        --input_csv <path> \
        --output_csv <path> \
        --mode dev|test \
        --batch_size 4
"""
# ── Unsloth 임포트 완전 차단 ──────────────────────────────────────────────────
import sys
_BLOCKED = {"unsloth", "unsloth_zoo"}
_real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

import builtins
_orig_import = builtins.__import__

def _blocking_import(name, *args, **kwargs):
    top = name.split(".")[0]
    if top in _BLOCKED:
        raise ImportError(f"[inference_worker] Unsloth import blocked: {name}")
    return _orig_import(name, *args, **kwargs)

builtins.__import__ = _blocking_import
# ─────────────────────────────────────────────────────────────────────────────

import os
import gc
import re
import json
import argparse

import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

MAX_SEQ_LENGTH = 2048
MAX_NEW_TOKENS = 192

QA_SYSTEM = (
    "당신은 한국어 대화 요약 전문가입니다.\n"
    "대화를 분석하여 화자들이 무엇을 논의하고 어떤 결정을 내렸는지 요약하세요.\n"
    "#Person1#, #Person2# 등의 화자 표기를 그대로 유지하세요.\n"
    "간결하게 1~2문장으로 작성하세요."
)
QA_USER = "이 대화에서 무슨 일이 일어났나요?\n\n{dialogue}"


def postprocess(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"#\s*Person\s*(\d+)\s*#", r"#Person\1#", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^요약\s*:\s*", "", text).strip()
    return text if text else "빈 요약"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path",  required=True)
    parser.add_argument("--input_csv",  required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--mode",       default="test", choices=["dev", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    adapter_cfg    = json.load(open(os.path.join(args.lora_path, "adapter_config.json")))
    base_model_name = adapter_cfg["base_model_name_or_path"]
    print(f"[worker] base model : {base_model_name}")
    print(f"[worker] adapter    : {args.lora_path}")
    print(f"[worker] mode={args.mode}  batch_size={args.batch_size}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.lora_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model.eval()

    df = pd.read_csv(args.input_csv)

    # 전체 텍스트 사전 생성
    texts = []
    for _, row in df.iterrows():
        messages = [
            {"role": "system", "content": QA_SYSTEM},
            {"role": "user",   "content": QA_USER.format(dialogue=row["dialogue"])},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        texts.append(text)

    preds = []
    device = next(model.parameters()).device
    bs = args.batch_size
    total_batches = (len(texts) + bs - 1) // bs

    for i in tqdm(range(0, len(texts), bs), total=total_batches,
                  desc=f"{args.mode} inference (batch={bs})"):
        batch_texts = texts[i:i + bs]
        inputs = tokenizer(
            batch_texts, return_tensors="pt",
            padding=True, truncation=True, max_length=MAX_SEQ_LENGTH,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        for seq in out:
            summary = tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()
            preds.append(postprocess(summary))

    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    if args.mode == "test":
        out_df = pd.DataFrame({"fname": df["fname"], "summary": preds})
    else:
        out_df = df.copy()
        out_df["pred_summary"] = preds

    out_df.to_csv(args.output_csv, index=(args.mode == "test"))
    print(f"[worker] 저장 완료: {args.output_csv}  ({len(out_df):,}행)")


if __name__ == "__main__":
    main()
