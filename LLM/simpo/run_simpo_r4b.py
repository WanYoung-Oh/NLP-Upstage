"""
SimPO 학습 스크립트 — r4b_response_only_ckpt 기반
====================================================
r4b_response_only_ckpt(현재 최고 SFT, qa_style 52.0083)를 베이스로
SimPO preference learning → qa_style 추론.

실행:
    cd /data/ephemeral/home/NLP/LLM/simpo
    python run_simpo_r4b.py              # Step 1~3 전체
    python run_simpo_r4b.py --skip_merge # merge 이미 완료된 경우
    python run_simpo_r4b.py --infer_only # 학습 완료 후 추론만

단계:
    Step 1. r4b SFT LoRA → base 모델에 merge (fp16, ~28GB VRAM or CPU offload)
    Step 2. Merged 모델 + 새 LoRA로 SimPO 학습 (4bit + QLoRA, ~1~2시간)
    Step 3. SimPO 모델 + qa_style 프롬프트로 dev/test 추론

출력:
    outputs/simpo_r4b/lora_adapter/     — SimPO LoRA 어댑터
    outputs/simpo_r4b/_merged_sft/      — Merged SFT (Step 2 베이스)
    prediction/simpo_r4b_dev.csv        — dev 평가용
    prediction/simpo_r4b_test.csv       — 제출용
"""

import os, sys, gc, re, json, argparse
from pathlib import Path

import torch
import pandas as pd
from tqdm.auto import tqdm

# ── 경로 ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
LLM_DIR    = SCRIPT_DIR.parent
ROOT_DIR   = LLM_DIR.parent

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(LLM_DIR))

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────
BASE_MODEL     = "unsloth/qwen3-14b-unsloth-bnb-4bit"
SFT_ADAPTER    = str(LLM_DIR / "mbr_ensemble" / "r4b_response_only_ckpt")
SIMPO_OUTPUT   = str(SCRIPT_DIR / "outputs" / "simpo_r4b")
MERGED_PATH    = os.path.join(SIMPO_OUTPUT, "_merged_sft")
ADAPTER_PATH   = os.path.join(SIMPO_OUTPUT, "lora_adapter")
DATA_FILE      = str(SCRIPT_DIR / "data" / "train_with_rejects.json")
DATA_DIR       = str(SCRIPT_DIR / "data")
PRED_DIR       = str(ROOT_DIR / "prediction")

MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 128

SIMPO_LORA_R      = 64
SIMPO_LORA_ALPHA  = 64
SIMPO_LR          = 3e-7
SIMPO_EPOCHS      = 1
SIMPO_BATCH       = 2
SIMPO_GRAD_ACCUM  = 4
SIMPO_GAMMA       = 1.0   # SimPO margin
SIMPO_BETA        = 5.0   # CPOTrainer beta

# qa_style 프롬프트 (LB 52.0083 기준 최고)
QA_SYSTEM = (
    "당신은 한국어 대화 요약 전문가입니다.\n"
    "대화를 분석하여 화자들이 무엇을 논의하고 어떤 결정을 내렸는지 요약하세요.\n"
    "#Person1#, #Person2# 등의 화자 표기를 그대로 유지하세요.\n"
    "간결하게 1~2문장으로 작성하세요."
)
QA_USER = "이 대화에서 무슨 일이 일어났나요?\n\n{dialogue}"


# ── 공통 유틸 ────────────────────────────────────────────────────────────────

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


# ── Step 1: SFT LoRA merge ────────────────────────────────────────────────────

def step1_merge():
    """r4b_response_only_ckpt를 base 모델에 merge → MERGED_PATH 저장."""
    weight_file = os.path.join(MERGED_PATH, "model.safetensors")
    weight_shard = os.path.join(MERGED_PATH, "model.safetensors.index.json")
    if os.path.exists(weight_file) or os.path.exists(weight_shard):
        print(f"[Step 1] Merged 모델 이미 존재: {MERGED_PATH} → 건너뜀")
        return

    print(f"\n{'='*60}")
    print("[Step 1] SFT LoRA → base 모델 merge")
    print(f"  SFT adapter : {SFT_ADAPTER}")
    print(f"  저장 경로   : {MERGED_PATH}")
    print(f"{'='*60}")

    os.makedirs(MERGED_PATH, exist_ok=True)

    from unsloth import FastLanguageModel

    # Unsloth가 adapter_config.json을 읽어 base + adapter를 한 번에 로드
    print("  Unsloth로 r4b adapter 로드 중...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SFT_ADAPTER,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        device_map={"": 0},
    )

    # Unsloth 전용 merge 저장 (4-bit → fp16 내부 처리, save_pretrained 오류 회피)
    print(f"  fp16 merge 저장 중: {MERGED_PATH}")
    model.save_pretrained_merged(MERGED_PATH, tokenizer, save_method="merged_16bit")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("[Step 1] 완료")


# ── Step 2: SimPO 학습 ────────────────────────────────────────────────────────

def step2_simpo():
    """SFT_ADAPTER(r4b) 위에 SimPO LoRA 학습. fp16 merge 없이 4-bit 직접 로드."""
    if os.path.exists(os.path.join(ADAPTER_PATH, "adapter_model.safetensors")):
        print(f"[Step 2] SimPO adapter 이미 존재: {ADAPTER_PATH} → 건너뜀")
        return

    print(f"\n{'='*60}")
    print("[Step 2] SimPO 학습")
    print(f"  베이스      : {BASE_MODEL} (pre-quantized 4bit)")
    print(f"  데이터      : {DATA_FILE}")
    print(f"  r={SIMPO_LORA_R}  lr={SIMPO_LR}  epoch={SIMPO_EPOCHS}")
    print(f"  gamma={SIMPO_GAMMA}  beta={SIMPO_BETA}")
    print(f"{'='*60}")

    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import CPOConfig, CPOTrainer
    from datasets import Dataset

    # BASE_MODEL 직접 로드 (pre-quantized 4bit, ~7-8GB, 어댑터 스택 문제 없음)
    # SFT_ADAPTER 로드 시 Unsloth가 기존 r4b 어댑터(r=32)와 충돌 → BASE_MODEL 사용
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        device_map={"": 0},
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=SIMPO_LORA_R,
        lora_alpha=SIMPO_LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # 데이터 준비
    raw = json.load(open(DATA_FILE))
    print(f"  학습 데이터: {len(raw)}쌍")

    dataset = Dataset.from_list([
        {
            "prompt": tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": QA_SYSTEM},
                    {"role": "user",   "content": QA_USER.format(dialogue=d["dialogue"])},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ),
            "chosen":   d["chosen"],
            "rejected": d["rejected"],
        }
        for d in raw
    ])

    os.makedirs(SIMPO_OUTPUT, exist_ok=True)
    trainer = CPOTrainer(
        model=model,
        args=CPOConfig(
            per_device_train_batch_size=SIMPO_BATCH,
            gradient_accumulation_steps=SIMPO_GRAD_ACCUM,
            num_train_epochs=SIMPO_EPOCHS,
            max_steps=500,
            learning_rate=SIMPO_LR,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            output_dir=SIMPO_OUTPUT,
            logging_steps=50,
            save_steps=500,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            seed=42,
            report_to="none",
            loss_type="simpo",
            cpo_alpha=0.0,
            simpo_gamma=SIMPO_GAMMA,
            beta=SIMPO_BETA,
            max_length=MAX_SEQ_LENGTH,
            max_prompt_length=MAX_SEQ_LENGTH - MAX_NEW_TOKENS,
        ),
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("\n  SimPO 학습 시작...")
    trainer.train()

    print(f"\n  adapter 저장: {ADAPTER_PATH}")
    model.save_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    print("[Step 2] 완료")


# ── 공통 추론 함수 ────────────────────────────────────────────────────────────

def _load_model_for_infer(base_path: str, adapter_path: str):
    """adapter_path(LoRA) 기준으로 Unsloth FastLanguageModel 로드.
    AutoModelForCausalLM 대신 FastLanguageModel 사용 → apply_qkv 패치 보장."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,   # adapter_config.json이 base 경로를 자동 참조
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        device_map={"": 0},
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def _run_infer(model, tokenizer, label: str, pred_prefix: str):
    """dev + test 추론 공통 실행."""
    os.makedirs(PRED_DIR, exist_ok=True)

    def infer_csv(input_csv, output_csv, mode):
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
        bs = 1   # Unsloth fast_forward_inference 배치 패딩 shape 충돌 우회
        device = next(model.parameters()).device
        for i in tqdm(range(0, len(texts), bs), desc=f"[{label}/{mode}]"):
            batch = texts[i:i+bs]
            inputs = tokenizer(
                batch, return_tensors="pt",
                padding=True, truncation=True, max_length=MAX_SEQ_LENGTH,
            ).to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=False,  # Unsloth fast_forward_inference RoPE shape 버그 우회
                    pad_token_id=tokenizer.eos_token_id,
                )
            input_len = inputs["input_ids"].shape[1]
            for seq in out:
                preds.append(postprocess(
                    tokenizer.decode(seq[input_len:], skip_special_tokens=True)
                ))

        if mode == "dev":
            out_df = df.copy()
            out_df["pred_summary"] = preds
            if "summary" in df.columns:
                print(f"\n[{label} dev ROUGE]")
                compute_rouge(preds, df["summary"].tolist())
        else:
            out_df = pd.DataFrame({"fname": df["fname"], "summary": preds})

        out_df.to_csv(output_csv, index=False)
        print(f"  저장: {output_csv}")

    infer_csv(
        os.path.join(DATA_DIR, "test.csv"),
        os.path.join(PRED_DIR, f"{pred_prefix}_test.csv"),
        "test",
    )


# ── Step 0: baseline_ckpt + qa_style 빠른 검증 ───────────────────────────────

def step0_eval_baseline():
    """재학습 없이 기존 baseline_ckpt + qa_style로 dev/test 추론.
    r4b qa_style(52.0) 대비 SimPO 효과 확인용."""
    baseline_adapter = str(SCRIPT_DIR / "outputs" / "baseline_ckpt")

    print(f"\n{'='*60}")
    print("[Step 0] baseline_ckpt + qa_style 추론 (재학습 없음)")
    print(f"  base   : {BASE_MODEL}")
    print(f"  adapter: {baseline_adapter}")
    print(f"{'='*60}")

    model, tokenizer = _load_model_for_infer(BASE_MODEL, baseline_adapter)
    _run_infer(model, tokenizer, label="baseline", pred_prefix="simpo_baseline_qa")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("[Step 0] 완료 — prediction/simpo_baseline_qa_dev.csv 확인 후 제출 여부 결정")


# ── Step 3: 추론 ──────────────────────────────────────────────────────────────

def step3_infer():
    """SimPO r4b adapter + qa_style 프롬프트로 dev/test 추론."""
    print(f"\n{'='*60}")
    print("[Step 3] SimPO r4b + qa_style 추론")
    print(f"  base   : {BASE_MODEL}")
    print(f"  adapter: {ADAPTER_PATH}")
    print(f"{'='*60}")

    model, tokenizer = _load_model_for_infer(BASE_MODEL, ADAPTER_PATH)
    _run_infer(model, tokenizer, label="simpo_r4b", pred_prefix="simpo_r4b")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("[Step 3] 완료")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_baseline", action="store_true",
                        help="재학습 없이 baseline_ckpt + qa_style로 dev/test 추론만 실행")
    parser.add_argument("--skip_merge", action="store_true",
                        help="Step 1(merge) 건너뜀 — MERGED_PATH 이미 존재할 때")
    parser.add_argument("--skip_train", action="store_true",
                        help="Step 2(SimPO 학습) 건너뜀 — adapter 이미 존재할 때")
    parser.add_argument("--infer_only", action="store_true",
                        help="Step 3(추론)만 실행")
    args = parser.parse_args()

    if args.infer_only:
        step3_infer()
        return

    if args.eval_baseline:
        step0_eval_baseline()
        return

    if not args.skip_merge:
        step1_merge()

    if not args.skip_train:
        step2_simpo()

    step3_infer()


if __name__ == "__main__":
    main()
