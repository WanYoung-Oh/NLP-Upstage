"""
Qwen3.5-35B-A3B 베이스 모델 추론 (SFT 없음)
=============================================
- 모델: unsloth/Qwen3.5-35B-A3B (MoE, 활성 파라미터 3B)
- 어댑터: 없음 (베이스 그대로)
- 프롬프트: qa_style (r4b 52.0083 달성 기준)
- dev → ROUGE 평가, test → 제출용 CSV

실행:
    cd /data/ephemeral/home/NLP/LLM/response_only_SFT
    python infer_qwen35_35b_a3b_base.py
"""

import os, gc, re, sys, time
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
LLM_DIR    = SCRIPT_DIR.parent
ROOT_DIR   = LLM_DIR.parent

for p in [str(LLM_DIR), str(ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── 설정 ──────────────────────────────────────────────────────────────────────
MODEL_NAME     = "unsloth/Qwen3.5-35B-A3B"
MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 128
DATA_DIR       = str(ROOT_DIR / "data")
PRED_DIR       = str(ROOT_DIR / "prediction")
LABEL          = "qwen35_35b_a3b_base"

# qa_style 프롬프트 (r4b 52.0083 기준)
QA_SYSTEM = (
    "당신은 한국어 대화 요약 전문가입니다.\n"
    "대화를 분석하여 화자들이 무엇을 논의하고 어떤 결정을 내렸는지 요약하세요.\n"
    "#Person1#, #Person2# 등의 화자 표기를 그대로 유지하세요.\n"
    "간결하게 1~2문장으로 작성하세요."
)
QA_USER = "이 대화에서 무슨 일이 일어났나요?\n\n{dialogue}"


# ── GPU 대기 ──────────────────────────────────────────────────────────────────
def wait_for_gpu(free_gb: float = 20.0, poll_sec: int = 60):
    if not torch.cuda.is_available():
        return
    while True:
        f, t = torch.cuda.mem_get_info(0)
        f_gb = f / 1e9
        print(f"[GPU 대기] 여유: {f_gb:.1f} / {t/1e9:.1f} GB (기준: {free_gb} GB)")
        if f_gb >= free_gb:
            break
        time.sleep(poll_sec)
    print("[GPU 대기] 완료")


# ── 후처리 ───────────────────────────────────────────────────────────────────
def postprocess(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"#\s*Person\s*(\d+)\s*#", r"#Person\1#", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(요약\s*:\s*|Summary\s*:\s*)", "", text).strip()
    return text if text else "빈 요약"


# ── ROUGE ─────────────────────────────────────────────────────────────────────
def compute_rouge(preds, golds):
    from rouge import Rouge
    rouge = Rouge()
    try:
        from prompts.mecab_ko import get_mecab
        m = get_mecab()
        pt = [" ".join(m.morphs(p)) if p.strip() else "빈요약" for p in preds]
        gt = [" ".join(m.morphs(g)) if g.strip() else "빈요약" for g in golds]
        method = "mecab"
    except Exception:
        pt, gt, method = preds, golds, "whitespace"
    s = rouge.get_scores(pt, gt, avg=True)
    r1, r2, rl = s["rouge-1"]["f"], s["rouge-2"]["f"], s["rouge-l"]["f"]
    print(f"  R1={r1:.4f}  R2={r2:.4f}  RL={rl:.4f}  Combined={r1+r2+rl:.4f}  [{method}]")
    return r1 + r2 + rl


# ── 모델 로드 ─────────────────────────────────────────────────────────────────
def load_model():
    import unsloth  # noqa
    from unsloth import FastLanguageModel

    print(f"\n[로드] {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        device_map={"": 0},
    )
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ── 추론 공통 ─────────────────────────────────────────────────────────────────
def run_infer(model, tokenizer, input_csv, output_csv, mode):
    df = pd.read_csv(input_csv)
    from src.data.preprocess import clean_text
    df["dialogue"] = df["dialogue"].apply(clean_text)

    preds = []
    device = next(model.parameters()).device
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"[{mode}]"):
        msg = [
            {"role": "system", "content": QA_SYSTEM},
            {"role": "user",   "content": QA_USER.format(dialogue=row["dialogue"])},
        ]
        text = tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
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
            print(f"\n[dev ROUGE — {LABEL}]")
            compute_rouge(preds, df["summary"].tolist())
    else:
        out_df = pd.DataFrame({"fname": df["fname"], "summary": preds})

    out_df.to_csv(output_csv, index=False)
    print(f"  저장: {output_csv}")


# ── 메인 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    wait_for_gpu(free_gb=20.0, poll_sec=60)

    model, tokenizer = load_model()

    run_infer(model, tokenizer,
              input_csv=os.path.join(DATA_DIR, "dev.csv"),
              output_csv=os.path.join(PRED_DIR, f"{LABEL}_dev.csv"),
              mode="dev")

    run_infer(model, tokenizer,
              input_csv=os.path.join(DATA_DIR, "test.csv"),
              output_csv=os.path.join(PRED_DIR, f"{LABEL}_test.csv"),
              mode="test")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n=== 완료 ===")
    print(f"  dev : {PRED_DIR}/{LABEL}_dev.csv")
    print(f"  test: {PRED_DIR}/{LABEL}_test.csv")
