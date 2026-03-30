"""
simpo_r4b/lora_adapter 체크포인트 추론 스크립트
- 베이스: unsloth/qwen3-14b-unsloth-bnb-4bit
- 어댑터: outputs/simpo_r4b/lora_adapter
- 토크나이저: 어댑터 폴더 내 저장본 (학습 시와 동일)
- 데이터: data/dev.csv, data/test.csv
"""

import os, gc, re, sys
from pathlib import Path

import torch
import pandas as pd
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent.parent

ADAPTER_PATH  = str(SCRIPT_DIR / "outputs" / "simpo_r4b" / "lora_adapter")
DATA_DIR      = str(SCRIPT_DIR / "data")
PRED_DIR      = str(ROOT_DIR / "prediction")

MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 128

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
    text = re.sub(r"^(요약\s*:\s*|Summary\s*:\s*)", "", text).strip()
    return text if text else "빈 요약"


def compute_rouge(preds, golds):
    from rouge import Rouge
    rouge = Rouge()
    try:
        sys.path.insert(0, str(ROOT_DIR))
        from prompts.mecab_ko import get_mecab
        m = get_mecab()
        preds_t = [" ".join(m.morphs(p)) if p.strip() else "빈요약" for p in preds]
        golds_t = [" ".join(m.morphs(g)) if g.strip() else "빈요약" for g in golds]
        method = "mecab"
    except Exception:
        preds_t = [p if p.strip() else "빈요약" for p in preds]
        golds_t = [g if g.strip() else "빈요약" for g in golds]
        method = "whitespace"
    s = rouge.get_scores(preds_t, golds_t, avg=True)
    r1, r2, rl = s["rouge-1"]["f"], s["rouge-2"]["f"], s["rouge-l"]["f"]
    print(f"  R1={r1:.4f}  R2={r2:.4f}  RL={rl:.4f}  Combined={r1+r2+rl:.4f}  [{method}]")
    return r1 + r2 + rl


def load_model():
    import unsloth  # noqa
    from unsloth import FastLanguageModel
    from peft import PeftModel
    from transformers import AutoTokenizer

    print(f"[load] 베이스 모델 로드: unsloth/qwen3-14b-unsloth-bnb-4bit")
    base_model, _ = FastLanguageModel.from_pretrained(
        model_name="unsloth/qwen3-14b-unsloth-bnb-4bit",
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        device_map={"": 0},
    )

    print(f"[load] LoRA 어댑터 로드: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    # 학습 시 저장된 토크나이저를 그대로 사용
    print(f"[load] 토크나이저 로드: {ADAPTER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


def infer_csv(model, tokenizer, input_csv, output_csv, mode):
    df = pd.read_csv(input_csv)
    texts = []
    for _, row in df.iterrows():
        msg = [
            {"role": "system", "content": QA_SYSTEM},
            {"role": "user",   "content": QA_USER.format(dialogue=row["dialogue"])},
        ]
        texts.append(tokenizer.apply_chat_template(
            msg, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        ))

    preds = []
    device = next(model.parameters()).device
    for i in tqdm(range(len(texts)), desc=f"[{mode}]"):
        inputs = tokenizer(
            [texts[i]], return_tensors="pt",
            padding=True, truncation=True, max_length=MAX_SEQ_LENGTH,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        preds.append(postprocess(
            tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        ))

    os.makedirs(PRED_DIR, exist_ok=True)
    if mode == "dev":
        out_df = df.copy()
        out_df["pred_summary"] = preds
        if "summary" in df.columns:
            print(f"\n[dev ROUGE]")
            compute_rouge(preds, df["summary"].tolist())
    else:
        out_df = pd.DataFrame({"fname": df["fname"], "summary": preds})

    out_df.to_csv(output_csv, index=False)
    print(f"  저장: {output_csv}")


def main():
    model, tokenizer = load_model()

    infer_csv(
        model, tokenizer,
        input_csv=os.path.join(DATA_DIR, "dev.csv"),
        output_csv=os.path.join(PRED_DIR, "simpo_r4b_lora_adapter_dev.csv"),
        mode="dev",
    )
    infer_csv(
        model, tokenizer,
        input_csv=os.path.join(DATA_DIR, "test.csv"),
        output_csv=os.path.join(PRED_DIR, "simpo_r4b_lora_adapter_test.csv"),
        mode="test",
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("\n완료.")


if __name__ == "__main__":
    main()
