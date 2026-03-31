"""
Train+Dev 병합 SFT — r4b 동일 설정 (Qwen3-14B)
================================================
r4b_response_only_ckpt(52.0083) 기반 설정을 그대로 사용,
train.csv + dev.csv를 모두 학습 데이터로 활용 → test.csv 추론.

실행:
    cd /data/ephemeral/home/NLP/LLM/response_only_SFT
    python run_r4b_traindev.py

출력:
    outputs/r4b_traindev_ckpt/lora_adapter/   — LoRA 어댑터
    ../../prediction/r4b_traindev_test.csv     — 제출용 test 추론 결과
"""

import os, gc, re, sys, random, time, dataclasses
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import unsloth  # noqa: must be before transformers
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

# ── 경로 ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
LLM_DIR    = SCRIPT_DIR.parent
ROOT_DIR   = LLM_DIR.parent

for p in [str(LLM_DIR), str(ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── 하이퍼파라미터 (r4b 동일) ──────────────────────────────────────────────────
BASE_MODEL       = "unsloth/qwen3-14b-unsloth-bnb-4bit"
EXP_NAME         = "r4b_traindev_ckpt"
LORA_R           = 32
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.0
LEARNING_RATE    = 2e-4
EPOCHS           = 2
WARMUP_RATIO     = 0.05
WEIGHT_DECAY     = 0.01
PER_DEVICE_BATCH = 2
GRAD_ACCUM       = 16     # effective batch = 32
MAX_SEQ_LENGTH   = 1024
MAX_NEW_TOKENS   = 128
SEED             = 5307

OUTPUT_DIR  = str(SCRIPT_DIR / "outputs" / EXP_NAME)
LORA_PATH   = os.path.join(OUTPUT_DIR, "lora_adapter")
DATA_PATH   = str(ROOT_DIR / "data")
PRED_DIR    = str(ROOT_DIR / "prediction")

# ── 프롬프트 ──────────────────────────────────────────────────────────────────
# 학습: base 프롬프트 (r4b_response_only_ckpt 학습 시 동일)
TRAIN_SYSTEM = (
    "당신은 한국어 대화 요약 전문가입니다.\n"
    "대화에는 #Person1#, #Person2# 등의 화자 태그가 사용됩니다.\n"
    "요약할 때 이 화자 태그를 그대로 사용하여 누가 무엇을 했는지 명확히 구분해주세요.\n"
    "핵심 내용만 1~3문장으로 간결하게 요약하세요."
)
TRAIN_USER = (
    "아래 대화를 읽고 핵심 내용을 요약해주세요.\n"
    "화자 태그(#Person1# 등)를 유지하세요.\n\n{dialogue}"
)

# 추론: qa_style 프롬프트 (r4b 52.0083 달성)
INFER_SYSTEM = (
    "당신은 한국어 대화 요약 전문가입니다.\n"
    "대화를 분석하여 화자들이 무엇을 논의하고 어떤 결정을 내렸는지 요약하세요.\n"
    "#Person1#, #Person2# 등의 화자 표기를 그대로 유지하세요.\n"
    "간결하게 1~2문장으로 작성하세요."
)
INFER_USER = "이 대화에서 무슨 일이 일어났나요?\n\n{dialogue}"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── GPU 대기 ──────────────────────────────────────────────────────────────────
def wait_for_gpu(free_gb_threshold: float = 18.0, poll_sec: int = 60):
    """GPU 여유 메모리가 threshold 이상이 될 때까지 대기.
    mem_get_info()는 시스템 전체 여유 메모리를 반환 (타 프로세스 포함)."""
    if not torch.cuda.is_available():
        return
    while True:
        free, total = torch.cuda.mem_get_info(0)
        free_gb = free / 1e9
        total_gb = total / 1e9
        print(f"[GPU 대기] 여유: {free_gb:.1f} / {total_gb:.1f} GB (기준: {free_gb_threshold} GB)")
        if free_gb >= free_gb_threshold:
            break
        time.sleep(poll_sec)
    print("[GPU 대기] 완료 — 학습 시작")


# ── 전처리 ───────────────────────────────────────────────────────────────────
def load_and_preprocess(csv_path: str, apply_filter: bool = True) -> pd.DataFrame:
    from src.data.preprocess import clean_text, filter_by_length
    df = pd.read_csv(csv_path)
    before = len(df)
    df["dialogue"] = df["dialogue"].apply(clean_text)
    if "summary" in df.columns:
        df["summary"] = df["summary"].apply(clean_text)
    if apply_filter and "summary" in df.columns:
        df = filter_by_length(df, dialogue_max=1500, summary_min=50, summary_max=250)
    after = len(df)
    print(f"[전처리] {os.path.basename(csv_path)}: {before} → {after}행")
    return df.reset_index(drop=True)


# ── ResponseOnlyDataCollator ──────────────────────────────────────────────────
@dataclasses.dataclass
class ResponseOnlyDataCollator:
    tokenizer: Any
    response_template_ids: List[int] = None
    max_length: int = 1024

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        if "input_ids" in features[0]:
            batch = self.tokenizer.pad(
                features, padding=True, max_length=self.max_length, return_tensors="pt"
            )
        else:
            texts = [f["text"] for f in features]
            batch = self.tokenizer(
                texts, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length,
            )
        labels = batch["input_ids"].clone()
        for i in range(len(labels)):
            ids = batch["input_ids"][i].tolist()
            pos = self._find_response_start(ids)
            if pos >= 0:
                labels[i, :pos] = -100
            labels[i, batch["attention_mask"][i] == 0] = -100
        batch["labels"] = labels
        return batch

    def _find_response_start(self, ids: List[int]) -> int:
        template = self.response_template_ids
        if template is None:
            return 0
        last_pos = -1
        for i in range(len(ids) - len(template) + 1):
            if ids[i:i + len(template)] == template:
                last_pos = i + len(template)
        return last_pos


# ── 후처리 ───────────────────────────────────────────────────────────────────
def postprocess(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"#\s*Person\s*(\d+)\s*#", r"#Person\1#", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(요약\s*:\s*|Summary\s*:\s*)", "", text).strip()
    return text if text else "빈 요약"


# ── Step 1: 학습 ──────────────────────────────────────────────────────────────
def train():
    if os.path.exists(os.path.join(LORA_PATH, "adapter_model.safetensors")):
        print(f"[학습] 어댑터 이미 존재: {LORA_PATH} → 건너뜀")
        return

    print(f"\n{'='*70}")
    print(f"[학습] Train+Dev 병합 SFT — {EXP_NAME}")
    print(f"  base={BASE_MODEL}")
    print(f"  r={LORA_R}, alpha={LORA_ALPHA}, lr={LEARNING_RATE}, epochs={EPOCHS}")
    print(f"  batch={PER_DEVICE_BATCH}, grad_accum={GRAD_ACCUM} (effective={PER_DEVICE_BATCH*GRAD_ACCUM})")
    print(f"{'='*70}\n")

    # GPU 여유 대기 (이전 추론 프로세스가 종료될 때까지)
    wait_for_gpu(free_gb_threshold=20.0, poll_sec=60)

    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import SFTTrainer, SFTConfig

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        device_map={"": 0},
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"학습 파라미터: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # train + dev 병합
    train_df = load_and_preprocess(os.path.join(DATA_PATH, "train.csv"), apply_filter=True)
    dev_df   = load_and_preprocess(os.path.join(DATA_PATH, "dev.csv"),   apply_filter=True)
    merged_df = pd.concat([train_df, dev_df], ignore_index=True)
    merged_df = merged_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"병합 데이터: train {len(train_df):,} + dev {len(dev_df):,} = {len(merged_df):,}개")

    response_template_str = "<|im_start|>assistant\n"
    response_template_ids = tokenizer.encode(response_template_str, add_special_tokens=False)
    print(f"response_template ids={response_template_ids}")

    def formatting_prompts_func(examples):
        texts = []
        for dialogue, summary in zip(examples["dialogue"], examples["summary"]):
            messages = [
                {"role": "system",    "content": TRAIN_SYSTEM},
                {"role": "user",      "content": TRAIN_USER.format(dialogue=dialogue)},
                {"role": "assistant", "content": str(summary)},
            ]
            texts.append(tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=False, enable_thinking=False,
            ))
        return {"text": texts}

    train_dataset = Dataset.from_pandas(merged_df[["dialogue", "summary"]]).map(
        formatting_prompts_func, batched=True
    )

    collator = ResponseOnlyDataCollator(
        tokenizer=tokenizer,
        response_template_ids=response_template_ids,
        max_length=MAX_SEQ_LENGTH,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sft_config = SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=20,
        save_strategy="no",   # eval 없이 학습만
        seed=SEED,
        output_dir=OUTPUT_DIR,
        report_to="none",
        optim="adamw_8bit",
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=sft_config,
        data_collator=collator,
    )

    print(f"\n[학습 시작] {len(merged_df):,}개 샘플...")
    stats = trainer.train()
    t = stats.metrics.get("train_runtime", 0)
    print(f"학습 완료: {t:.0f}초 ({t/60:.1f}분)")

    model.save_pretrained(LORA_PATH)
    tokenizer.save_pretrained(LORA_PATH)
    print(f"저장 완료: {LORA_PATH}")

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


# ── Step 2: test 추론 ─────────────────────────────────────────────────────────
def infer_test():
    test_out = os.path.join(PRED_DIR, f"{EXP_NAME}_test.csv")
    if os.path.exists(test_out):
        print(f"[추론] 이미 존재: {test_out} → 건너뜀")
        return

    print(f"\n{'='*70}")
    print(f"[추론] test.csv — {EXP_NAME}")
    print(f"{'='*70}\n")

    from unsloth import FastLanguageModel
    from peft import PeftModel

    base_model, _ = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
    from src.data.preprocess import clean_text
    test_df["dialogue"] = test_df["dialogue"].apply(clean_text)

    preds = []
    device = next(model.parameters()).device
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="[test]"):
        msg = [
            {"role": "system", "content": INFER_SYSTEM},
            {"role": "user",   "content": INFER_USER.format(dialogue=row["dialogue"])},
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
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        preds.append(postprocess(
            tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        ))

    os.makedirs(PRED_DIR, exist_ok=True)
    pd.DataFrame({"fname": test_df["fname"], "summary": preds}).to_csv(test_out, index=False)
    print(f"\n저장 완료: {test_out}")

    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()


# ── 메인 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()
    infer_test()
    print("\n=== 전체 완료 ===")
    print(f"  어댑터  : {LORA_PATH}")
    print(f"  제출파일: {PRED_DIR}/{EXP_NAME}_test.csv")
