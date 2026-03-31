"""
Qwen3.5-9B Response-Only SFT QLoRA — 최종 실험 (train+dev 병합)
================================================================
기반: run_qlora_q35_b_v2.py

변경점:
  - train.csv + dev.csv 병합 학습 (eval 없음)
  - enable_thinking=True 모드
  - Topic을 USER_TEMPLATE에 포함하여 학습
  - LoRA R=32, alpha=32
  - SEED=3407, EPOCHS=3, MAX_NEW_TOKENS=128
  - 학습 완료 후 test.csv 추론 → prediction/ 저장

실행:
    cd /data/ephemeral/home/NLP/LLM/response_only_SFT
    python run_q35_final_traindev.py
"""

import os
import gc
import re
import sys
import random
import dataclasses
from typing import Any, Dict, List

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import unsloth  # noqa: F401 — transformers보다 먼저 임포트해야 최적화 적용됨
from unsloth import FastModel, is_bfloat16_supported

import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_DIR    = os.path.dirname(SCRIPT_DIR)
ROOT_DIR   = os.path.dirname(LLM_DIR)

for p in [LLM_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ──────────────────────────────────────────────────────────────────────────────
# 재현성
# ──────────────────────────────────────────────────────────────────────────────
SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ──────────────────────────────────────────────────────────────────────────────
# 하이퍼파라미터
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAME       = "Qwen/Qwen3.5-9B"
EXP_NAME         = "q35_final_traindev_r32_a32"
MAX_SEQ_LENGTH   = 2048
LORA_R           = 32
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.0
LEARNING_RATE    = 2e-4
EPOCHS           = 3
WARMUP_RATIO     = 0.05
WEIGHT_DECAY     = 0.01
PER_DEVICE_BATCH = 2
GRAD_ACCUM       = 16    # effective batch = 2 × 16 = 32
MAX_NEW_TOKENS   = 128
DATA_PATH        = os.path.join(SCRIPT_DIR, "data")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs", EXP_NAME)
LORA_PATH  = os.path.join(OUTPUT_DIR, "lora_adapter")
PRED_DIR   = os.path.join(ROOT_DIR, "prediction")

# ──────────────────────────────────────────────────────────────────────────────
# 프롬프트 — Topic 포함, enable_thinking=True
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "당신은 한국어 대화 요약 전문가입니다. "
    "대화에는 #Person1#, #Person2# 등의 화자 태그가 사용됩니다. "
    "요약할 때 이 화자 태그를 그대로 사용하여 누가 무엇을 했는지 명확히 구분해주세요. "
    "핵심 내용만 1~3문장으로 간결하게 요약하세요. "
    "대화에 등장하는 사람 이름, 장소, 제품명 등 고유명사는 원문 그대로(영어면 영어, 한글이면 한글) 사용하세요."
)

# 학습용: Topic 포함
USER_TEMPLATE_TRAIN = (
    "아래 대화의 주제는 \"{topic}\"입니다.\n"
    "대화를 읽고 핵심 내용을 요약해주세요. "
    "화자 태그(#Person1# 등)를 유지하세요.\n\n{dialogue}"
)

# 추론용: test.csv에 topic 없으므로 topic 미포함
USER_TEMPLATE_INFER = (
    "아래 대화를 읽고 핵심 내용을 요약해주세요. "
    "화자 태그(#Person1# 등)를 유지하세요.\n\n{dialogue}"
)


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 전처리
# ──────────────────────────────────────────────────────────────────────────────
def load_and_preprocess(csv_path: str, is_train: bool = True) -> pd.DataFrame:
    from src.data.preprocess import clean_text, filter_by_length

    df = pd.read_csv(csv_path)
    before = len(df)

    df["dialogue"] = df["dialogue"].apply(clean_text)
    if "summary" in df.columns:
        df["summary"] = df["summary"].apply(clean_text)

    if is_train and "summary" in df.columns:
        df = filter_by_length(
            df,
            dialogue_max=1500,
            summary_min=20,
            summary_max=250,
        )

    after = len(df)
    print(f"[전처리] {os.path.basename(csv_path)}: {before} → {after}행")
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# ResponseOnlyDataCollator
# ──────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class ResponseOnlyDataCollator:
    tokenizer: Any
    response_template_ids: List[int] = None
    max_length: int = 2048

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        if "input_ids" in features[0]:
            batch = self.tokenizer.pad(
                features, padding=True,
                max_length=self.max_length, return_tensors="pt"
            )
        else:
            texts = [f["text"] for f in features]
            batch = self.tokenizer(
                texts, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length
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
# 메인
# ──────────────────────────────────────────────────────────────────────────────
def main():
    import time
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"실험: {EXP_NAME}")
    print(f"  model={MODEL_NAME}")
    print(f"  lora_r={LORA_R}, lora_alpha={LORA_ALPHA}, lr={LEARNING_RATE}")
    print(f"  epochs={EPOCHS}, seed={SEED}, max_new_tokens={MAX_NEW_TOKENS}")
    print(f"  batch={PER_DEVICE_BATCH}, grad_accum={GRAD_ACCUM} (effective={PER_DEVICE_BATCH*GRAD_ACCUM})")
    print(f"  enable_thinking=True, topic 포함 학습")
    print(f"  학습 데이터: train.csv + dev.csv 병합")
    print(f"{'='*80}\n")

    skip_training = os.path.exists(os.path.join(LORA_PATH, "adapter_model.safetensors"))

    # ── 1. 학습 ───────────────────────────────────────────────────────────────
    if not skip_training:
        # 모델 로드 (4bit 우선, 실패 시 8bit)
        try:
            model, tokenizer = FastModel.from_pretrained(
                model_name=MODEL_NAME,
                max_seq_length=MAX_SEQ_LENGTH,
                load_in_4bit=True,
                load_in_8bit=False,
                full_finetuning=False,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            print("모델 로드 완료 (4bit)")
        except Exception as e:
            print(f"4bit 로드 실패 ({e}), 8bit로 재시도...")
            model, tokenizer = FastModel.from_pretrained(
                model_name=MODEL_NAME,
                max_seq_length=MAX_SEQ_LENGTH,
                load_in_4bit=False,
                load_in_8bit=True,
                full_finetuning=False,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            print("모델 로드 완료 (8bit)")

        # Qwen3.5 VL Processor → tokenizer 추출
        if hasattr(tokenizer, "tokenizer"):
            tokenizer = tokenizer.tokenizer

        # LoRA 설정
        model = FastModel.get_peft_model(
            model,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=SEED,
            use_rslora=False,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"학습 파라미터: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        # 데이터 준비: train + dev 병합
        train_df = load_and_preprocess(os.path.join(DATA_PATH, "train.csv"), is_train=True)
        dev_df   = load_and_preprocess(os.path.join(DATA_PATH, "dev.csv"),   is_train=True)
        merged_df = pd.concat([train_df, dev_df], ignore_index=True)
        merged_df = merged_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        print(f"병합 학습 데이터: train {len(train_df):,} + dev {len(dev_df):,} = {len(merged_df):,}개")

        response_template_str = "<|im_start|>assistant\n"
        response_template_ids = tokenizer.encode(response_template_str, add_special_tokens=False)
        print(f"response_template: {tokenizer.decode(response_template_ids)!r}  ids={response_template_ids}")

        def formatting_prompts_func(examples):
            texts = []
            for dialogue, summary, topic in zip(
                examples["dialogue"], examples["summary"], examples["topic"]
            ):
                topic_str = str(topic) if pd.notna(topic) else ""
                messages = [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": USER_TEMPLATE_TRAIN.format(
                        topic=topic_str, dialogue=dialogue
                    )},
                    {"role": "assistant", "content": str(summary)},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False,
                    add_generation_prompt=False, enable_thinking=True,
                )
                texts.append(text)
            return {"text": texts}

        train_dataset = Dataset.from_pandas(
            merged_df[["dialogue", "summary", "topic"]]
        ).map(formatting_prompts_func, batched=True)

        collator = ResponseOnlyDataCollator(
            tokenizer=tokenizer,
            response_template_ids=response_template_ids,
            max_length=MAX_SEQ_LENGTH,
        )

        # Trainer 설정 (eval 없음 — train+dev 전체 학습)
        from trl import SFTTrainer, SFTConfig

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

            logging_steps=10,
            eval_strategy="no",
            save_strategy="epoch",
            save_total_limit=1,

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

        print(f"\n학습 시작...")
        t0 = time.time()
        stats = trainer.train()
        train_time = time.time() - t0
        print(f"학습 완료: {train_time:.0f}초 ({train_time/60:.1f}분)")

        model.save_pretrained(LORA_PATH)
        tokenizer.save_pretrained(LORA_PATH)
        print(f"LoRA 저장: {LORA_PATH}")

        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

    else:
        print(f"체크포인트 존재 → 학습 건너뜀: {LORA_PATH}")

    # ── 2. test.csv 추론 ──────────────────────────────────────────────────────
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    pred_path = os.path.join(PRED_DIR, f"q35_final_traindev_{timestamp}.csv")

    print(f"\ntest.csv 추론 시작 (enable_thinking=True, greedy)...")

    import json
    from peft import PeftModel
    adapter_cfg     = json.load(open(os.path.join(LORA_PATH, "adapter_config.json")))
    base_model_name = adapter_cfg["base_model_name_or_path"]
    print(f"  base model: {base_model_name}")

    base_model_inf, _ = FastModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    tokenizer_inf = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)
    tokenizer_inf.padding_side = "left"
    if tokenizer_inf.pad_token is None:
        tokenizer_inf.pad_token = tokenizer_inf.eos_token

    model_inf = PeftModel.from_pretrained(base_model_inf, LORA_PATH)
    model_inf.eval()
    model_inf.generation_config.max_length = MAX_SEQ_LENGTH

    test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
    print(f"테스트 데이터: {len(test_df)}행")

    preds = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="test inference"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_TEMPLATE_INFER.format(dialogue=row["dialogue"])},
        ]
        text = tokenizer_inf.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=True,
        )
        inputs = tokenizer_inf(
            text, return_tensors="pt",
            truncation=True, max_length=MAX_SEQ_LENGTH
        ).to(model_inf.device)
        with torch.no_grad():
            out = model_inf.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer_inf.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        summary   = tokenizer_inf.decode(generated, skip_special_tokens=True).strip()
        preds.append(postprocess(summary))

    result_df = test_df[["fname"]].copy()
    result_df["summary"] = preds
    result_df.to_csv(pred_path, index=False)
    print(f"\n추론 완료. 저장: {pred_path} ({len(result_df)}행)")

    del model_inf, base_model_inf
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
