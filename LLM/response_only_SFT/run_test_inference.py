"""
Test 추론 스크립트 — A/B 비교 후 최고 체크포인트로 test.csv 추론

사용법:
    cd /data/ephemeral/home/NLP/LLM/response_only_SFT
    python run_test_inference.py
    # 또는 특정 실험 지정:
    python run_test_inference.py --exp exp_A_r64_a128_lr2e4
"""

import os
import gc
import re
import sys
import json
import argparse
from typing import List

import pandas as pd
import torch
from tqdm.auto import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_DIR    = os.path.dirname(SCRIPT_DIR)
ROOT_DIR   = os.path.dirname(LLM_DIR)

for p in [LLM_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

DATA_PATH       = os.path.join(SCRIPT_DIR, "data")
PRED_PATH       = os.path.join(SCRIPT_DIR, "prediction")
MAX_SEQ_LENGTH  = 2048
MAX_NEW_TOKENS  = 192

# qa_style 프롬프트 (리더보드 최고점 기준)
QA_SYSTEM = (
    "당신은 한국어 대화 요약 전문가입니다.\n"
    "대화를 분석하여 화자들이 무엇을 논의하고 어떤 결정을 내렸는지 요약하세요.\n"
    "#Person1#, #Person2# 등의 화자 표기를 그대로 유지하세요.\n"
    "간결하게 1~2문장으로 작성하세요."
)
QA_USER = "이 대화에서 무슨 일이 일어났나요?\n\n{dialogue}"


# ──────────────────────────────────────────────────────────────────────────────
# 후처리
# ──────────────────────────────────────────────────────────────────────────────
def postprocess(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"#\s*Person\s*(\d+)\s*#", r"#Person\1#", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^요약\s*:\s*", "", text).strip()
    return text if text else "빈 요약"


# ──────────────────────────────────────────────────────────────────────────────
# ROUGE 평가 (dev 결과 비교용)
# ──────────────────────────────────────────────────────────────────────────────
def compute_rouge_combined(preds: List[str], golds: List[str]) -> dict:
    from rouge import Rouge
    rouge = Rouge()
    try:
        from prompts.mecab_ko import get_mecab
        m = get_mecab()
        preds_m = [" ".join(m.morphs(p)) if p.strip() else "빈요약" for p in preds]
        golds_m = [" ".join(m.morphs(g)) if g.strip() else "빈요약" for g in golds]
        scores = rouge.get_scores(preds_m, golds_m, avg=True)
        method = "mecab"
    except Exception:
        preds_s = [p if p.strip() else "빈요약" for p in preds]
        golds_s = [g if g.strip() else "빈요약" for g in golds]
        scores = rouge.get_scores(preds_s, golds_s, avg=True)
        method = "whitespace"
    r1 = scores["rouge-1"]["f"]
    r2 = scores["rouge-2"]["f"]
    rl = scores["rouge-l"]["f"]
    return {"r1": r1, "r2": r2, "rl": rl, "combined": r1 + r2 + rl, "method": method}


# ──────────────────────────────────────────────────────────────────────────────
# dev 추론 (누락된 경우 실행)
# ──────────────────────────────────────────────────────────────────────────────
def run_dev_inference_if_missing(exp_name: str):
    """dev 추론 CSV가 없으면 추론 실행 후 저장"""
    pred_file = os.path.join(PRED_PATH, f"dev_{exp_name}_qa_style.csv")
    if os.path.exists(pred_file):
        print(f"  [{exp_name}] dev 추론 파일 존재 → 스킵")
        return

    lora_path = os.path.join(
        os.path.dirname(PRED_PATH), "outputs", exp_name, "lora_adapter"
    )
    if not os.path.exists(os.path.join(lora_path, "adapter_model.safetensors")):
        print(f"  [{exp_name}] adapter 없음 → dev 추론 스킵")
        return

    print(f"\n  [{exp_name}] dev 추론 시작 (adapter 있음, pred 없음)...")
    adapter_cfg = json.load(open(os.path.join(lora_path, "adapter_config.json")))
    base_model_name = adapter_cfg["base_model_name_or_path"]
    model, tokenizer = _load_model_and_tokenizer(lora_path, base_model_name)

    dev_df = pd.read_csv(os.path.join(DATA_PATH, "dev.csv"))
    preds = []
    for _, row in tqdm(dev_df.iterrows(), total=len(dev_df),
                       desc=f"dev inference [{exp_name}]"):
        messages = [
            {"role": "system", "content": QA_SYSTEM},
            {"role": "user",   "content": QA_USER.format(dialogue=row["dialogue"])},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=MAX_SEQ_LENGTH
        ).to(next(model.parameters()).device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        summary = tokenizer.decode(generated, skip_special_tokens=True).strip()
        preds.append(postprocess(summary))

    os.makedirs(PRED_PATH, exist_ok=True)
    dev_df.assign(pred_summary=preds).to_csv(pred_file, index=False)
    print(f"  dev 추론 저장: {pred_file}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# A/B 비교 → 최고 실험 결정
# ──────────────────────────────────────────────────────────────────────────────
def pick_best_experiment(exp_names: List[str]) -> str:
    # 누락된 dev 추론 먼저 실행
    for name in exp_names:
        run_dev_inference_if_missing(name)

    dev_df = pd.read_csv(os.path.join(DATA_PATH, "dev.csv"))
    results = []
    for name in exp_names:
        pred_file = os.path.join(PRED_PATH, f"dev_{name}_qa_style.csv")
        if not os.path.exists(pred_file):
            print(f"[경고] {name} dev 추론 파일 없음 → 비교에서 제외")
            continue
        preds = pd.read_csv(pred_file)["pred_summary"].tolist()
        golds = dev_df["summary"].tolist()
        s = compute_rouge_combined(preds, golds)
        results.append({"name": name, **s})
        print(f"  {name}: Combined={s['combined']:.4f}  "
              f"(R1={s['r1']:.4f}, R2={s['r2']:.4f}, RL={s['rl']:.4f}) [{s['method']}]")

    if not results:
        raise RuntimeError("비교할 dev 추론 결과가 없습니다.")

    results.sort(key=lambda x: x["combined"], reverse=True)
    best = results[0]
    print(f"\n→ 최고 실험: {best['name']}  Combined={best['combined']:.4f}")
    return best["name"]


# ──────────────────────────────────────────────────────────────────────────────
# 추론 (표준 HF + PEFT — Unsloth fast kernel 우회)
# ──────────────────────────────────────────────────────────────────────────────
def _load_model_and_tokenizer(lora_path: str, base_model_name: str):
    """
    Unsloth FastLanguageModel로 base 로드 후 PEFT adapter 적용.
    - AutoModelForCausalLM 사용 시 Unsloth 전역 패치가 apply_qkv 없음 오류 유발
    - FastLanguageModel.for_inference() 미호출 (RoPE shape mismatch 버그 우회)
    """
    from unsloth import FastLanguageModel
    from peft import PeftModel

    print(f"  FastLanguageModel 로드: {base_model_name}")
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )
    print(f"  PEFT adapter 로드: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    # for_inference() 호출 금지 — Qwen3 RoPE shape mismatch 버그
    model.eval()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def run_inference(lora_path: str, test_df: pd.DataFrame, out_csv: str,
                  batch_size: int = 4):
    adapter_cfg = json.load(open(os.path.join(lora_path, "adapter_config.json")))
    base_model_name = adapter_cfg["base_model_name_or_path"]
    print(f"  base model : {base_model_name}")
    print(f"  adapter    : {lora_path}")
    print(f"  샘플 수    : {len(test_df):,}개  batch_size={batch_size}")

    model, tokenizer = _load_model_and_tokenizer(lora_path, base_model_name)

    # 전체 텍스트 준비
    texts = []
    for _, row in test_df.iterrows():
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
    total_batches = (len(texts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(texts), batch_size),
                  total=total_batches, desc="test inference (batch)"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors="pt",
            padding=True, truncation=True, max_length=MAX_SEQ_LENGTH
        )
        # device 이동
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        # 입력 길이만큼 제거
        input_len = inputs["input_ids"].shape[1]
        for seq in out:
            generated = seq[input_len:]
            summary = tokenizer.decode(generated, skip_special_tokens=True).strip()
            preds.append(postprocess(summary))

    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()

    # 제출 형식으로 저장
    submission = pd.DataFrame({
        "fname": test_df["fname"],
        "summary": preds,
    })
    submission.to_csv(out_csv, index=True)
    print(f"\n제출 파일 저장: {out_csv}  ({len(submission):,}행)")
    return preds


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp", type=str, default=None,
        help="사용할 실험 이름 (예: exp_A_r64_a128_lr2e4). 미지정 시 A/B 자동 비교."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="배치 추론 크기 (기본값: 4). OOM 발생 시 2 또는 1로 줄이세요."
    )
    args = parser.parse_args()

    os.makedirs(PRED_PATH, exist_ok=True)

    # ── 1. 최고 실험 결정 ────────────────────────────────────────────────────
    available_exps = ["exp_A_r64_a128_lr2e4", "exp_B_r32_a64_lr1e4"]

    if args.exp:
        best_exp = args.exp
        print(f"지정된 실험 사용: {best_exp}")
    else:
        print("=" * 60)
        print("A/B 실험 dev ROUGE 비교")
        print("=" * 60)
        best_exp = pick_best_experiment(available_exps)

    # ── 2. LoRA adapter 경로 확인 ────────────────────────────────────────────
    lora_path = os.path.join(SCRIPT_DIR, "outputs", best_exp, "lora_adapter")
    adapter_file = os.path.join(lora_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        raise FileNotFoundError(
            f"LoRA adapter 없음: {adapter_file}\n"
            f"실험 {best_exp}의 학습이 완료되지 않았습니다."
        )

    # ── 3. 이미 test 추론 완료 확인 ─────────────────────────────────────────
    out_csv = os.path.join(PRED_PATH, f"test_{best_exp}_qa_style.csv")
    if os.path.exists(out_csv):
        print(f"test 추론 결과 이미 존재: {out_csv}")
        print("재실행하려면 해당 파일을 삭제 후 재시도하세요.")
        return

    # ── 4. test 추론 ─────────────────────────────────────────────────────────
    test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
    print(f"\n[test 추론 시작]  실험: {best_exp}  샘플: {len(test_df):,}개")
    print("=" * 60)
    run_inference(lora_path, test_df, out_csv, batch_size=args.batch_size)
    print("\n완료!")


if __name__ == "__main__":
    main()
