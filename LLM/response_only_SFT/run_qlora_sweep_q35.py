"""
Qwen3.5-9B Response-Only SFT QLoRA 스윕
========================================
PLAN_3.5.md 기반 구현.

실험:
  q35_A: r=32, alpha=64,  lr=1e-4
  q35_B: r=64, alpha=128, lr=2e-4

주요 변경점 (run_qlora_sweep.py 대비):
  - FastModel (Qwen3.5 지원) + Qwen/Qwen3.5-9B
  - PER_DEVICE_BATCH=2, GRAD_ACCUM=16 (VRAM 여유 활용)
  - clean_text() + filter_by_length() 전처리
  - use_cache=False (Unsloth fast_forward_inference RoPE 버그 예방)

실행:
    cd /data/ephemeral/home/NLP/LLM/response_only_SFT
    python run_qlora_sweep_q35.py
"""

import os
import gc
import re
import sys
import random
import dataclasses
from typing import Any, Dict, List

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
import numpy as np
import torch
from datasets import Dataset
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
# 고정 하이퍼파라미터
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAME       = "Qwen/Qwen3.5-9B"
MAX_SEQ_LENGTH   = 2048
LORA_DROPOUT     = 0.0
EPOCHS           = 1
WARMUP_RATIO     = 0.05
WEIGHT_DECAY     = 0.01
PER_DEVICE_BATCH = 2     # Qwen3.5-9B: ~4.5GB 4-bit → VRAM 여유로 batch 확대
GRAD_ACCUM       = 16    # effective batch = 2 × 16 = 32 (기존과 동일)
MAX_NEW_TOKENS   = 192
DATA_PATH        = os.path.join(ROOT_DIR, "data")

# ──────────────────────────────────────────────────────────────────────────────
# 스윕 실험 정의
# ──────────────────────────────────────────────────────────────────────────────
EXPERIMENTS = [
    {"name": "q35_A_r32_a64_lr1e4",  "LORA_R": 32, "LORA_ALPHA": 64,  "LEARNING_RATE": 1e-4},
    {"name": "q35_B_r64_a128_lr2e4", "LORA_R": 64, "LORA_ALPHA": 128, "LEARNING_RATE": 2e-4},
]

# ──────────────────────────────────────────────────────────────────────────────
# 프롬프트 (qa_style — 리더보드 최고점 기준, r4b 52.0083)
# ──────────────────────────────────────────────────────────────────────────────
QA_SYSTEM = (
    "당신은 한국어 대화 요약 전문가입니다.\n"
    "대화를 분석하여 화자들이 무엇을 논의하고 어떤 결정을 내렸는지 요약하세요.\n"
    "#Person1#, #Person2# 등의 화자 표기를 그대로 유지하세요.\n"
    "간결하게 1~2문장으로 작성하세요."
)
QA_USER = "이 대화에서 무슨 일이 일어났나요?\n\n{dialogue}"


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 전처리 (src/data/preprocess.py 활용)
# ──────────────────────────────────────────────────────────────────────────────
def load_and_preprocess(csv_path: str, is_train: bool = True) -> pd.DataFrame:
    """clean_text + filter_by_length 적용."""
    from src.data.preprocess import clean_text, filter_by_length

    df = pd.read_csv(csv_path)
    before = len(df)

    df["dialogue"] = df["dialogue"].apply(clean_text)
    if "summary" in df.columns:
        df["summary"] = df["summary"].apply(clean_text)

    if is_train and "summary" in df.columns:
        df = filter_by_length(
            df,
            dialogue_max=830,
            summary_min=50,
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
# ROUGE 평가
# ──────────────────────────────────────────────────────────────────────────────
def compute_rouge_combined(preds: List[str], golds: List[str]) -> Dict:
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
# 단일 실험 실행
# ──────────────────────────────────────────────────────────────────────────────
def run_experiment(exp: Dict) -> Dict:
    exp_name   = exp["name"]
    LORA_R     = exp["LORA_R"]
    LORA_ALPHA = exp["LORA_ALPHA"]
    LR         = exp["LEARNING_RATE"]

    print(f"\n{'='*80}")
    print(f"실험 시작: {exp_name}")
    print(f"  model={MODEL_NAME}")
    print(f"  lora_r={LORA_R}, lora_alpha={LORA_ALPHA}, lr={LR}, epochs={EPOCHS}")
    print(f"  batch={PER_DEVICE_BATCH}, grad_accum={GRAD_ACCUM} (effective={PER_DEVICE_BATCH*GRAD_ACCUM})")
    print(f"{'='*80}\n")

    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs", exp_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lora_path = os.path.join(OUTPUT_DIR, "lora_adapter")

    from unsloth import FastModel

    skip_training  = os.path.exists(os.path.join(lora_path, "adapter_model.safetensors"))
    pred_path      = os.path.join(SCRIPT_DIR, f"prediction/dev_{exp_name}_qa_style.csv")
    skip_inference = os.path.exists(pred_path)
    train_time     = 0

    if not skip_training:
        # ── 1. 모델 로드 ──────────────────────────────────────────────────────
        model, tokenizer = FastModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
            dtype=torch.bfloat16,
        )

        # ── 2. LoRA 설정 ──────────────────────────────────────────────────────
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

        # ── 3. 데이터 준비 (전처리 포함) ──────────────────────────────────────
        train_df = load_and_preprocess(os.path.join(DATA_PATH, "train.csv"), is_train=True)
        dev_df   = load_and_preprocess(os.path.join(DATA_PATH, "dev.csv"),   is_train=False)
        print(f"Train {len(train_df):,}개 / Dev {len(dev_df):,}개")

        # response_template_ids 검증 출력
        response_template_str = "<|im_start|>assistant\n"
        response_template_ids = tokenizer.encode(response_template_str, add_special_tokens=False)
        print(f"response_template: {tokenizer.decode(response_template_ids)!r}  ids={response_template_ids}")

        def formatting_prompts_func(examples):
            texts = []
            for dialogue, summary in zip(examples["dialogue"], examples["summary"]):
                messages = [
                    {"role": "system",    "content": QA_SYSTEM},
                    {"role": "user",      "content": QA_USER.format(dialogue=dialogue)},
                    {"role": "assistant", "content": str(summary)},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False,
                    add_generation_prompt=False, enable_thinking=False,
                )
                texts.append(text)
            return {"text": texts}

        train_dataset = Dataset.from_pandas(train_df[["dialogue", "summary"]]).map(
            formatting_prompts_func, batched=True
        )
        dev_dataset = Dataset.from_pandas(dev_df[["dialogue", "summary"]]).map(
            formatting_prompts_func, batched=True
        )

        collator = ResponseOnlyDataCollator(
            tokenizer=tokenizer,
            response_template_ids=response_template_ids,
            max_length=MAX_SEQ_LENGTH,
        )

        # ── 4. Trainer 설정 ───────────────────────────────────────────────────
        from trl import SFTTrainer, SFTConfig
        from unsloth import is_bfloat16_supported

        steps_per_epoch = len(train_dataset) // (PER_DEVICE_BATCH * GRAD_ACCUM)
        eval_steps_val  = max(1, steps_per_epoch // 2)
        print(f"steps/epoch={steps_per_epoch}, eval_steps={eval_steps_val}")

        sft_config = SFTConfig(
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            packing=False,

            per_device_train_batch_size=PER_DEVICE_BATCH,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=GRAD_ACCUM,

            num_train_epochs=EPOCHS,
            learning_rate=LR,
            lr_scheduler_type="cosine",
            warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,

            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),

            logging_steps=10,
            eval_strategy="steps",
            eval_steps=eval_steps_val,
            save_strategy="steps",
            save_steps=eval_steps_val,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",

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
            eval_dataset=dev_dataset,
            args=sft_config,
            data_collator=collator,
        )

        # ── 5. 학습 ───────────────────────────────────────────────────────────
        print(f"\n[{exp_name}] 학습 시작...")
        stats = trainer.train()
        train_time = stats.metrics.get("train_runtime", 0)
        print(f"학습 완료: {train_time:.0f}초 ({train_time/60:.1f}분)")

        model.save_pretrained(lora_path)
        tokenizer.save_pretrained(lora_path)
        print(f"LoRA 저장: {lora_path}")
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

    else:
        print(f"[{exp_name}] 체크포인트 존재 → 학습 건너뜀, 추론만 실행")

    # ── 6. dev 추론 ───────────────────────────────────────────────────────────
    if skip_inference:
        print(f"[{exp_name}] 추론 결과 파일 존재 → 추론 건너뜀, ROUGE만 재계산")
        dev_df = load_and_preprocess(os.path.join(DATA_PATH, "dev.csv"), is_train=False)
        preds  = pd.read_csv(pred_path)["pred_summary"].tolist()
        golds  = dev_df["summary"].tolist()
        scores = compute_rouge_combined(preds, golds)
        print(f"\n{'='*60}")
        print(f"[{exp_name}] Dev ROUGE ({scores['method']} 기반, 캐시)")
        print(f"  R1={scores['r1']:.4f}  R2={scores['r2']:.4f}  RL={scores['rl']:.4f}  Combined={scores['combined']:.4f}")
        print(f"{'='*60}\n")
        return {"exp_name": exp_name, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
                "lr": LR, "train_sec": train_time, **scores}

    print(f"\n[{exp_name}] dev 추론 시작 (qa_style, greedy, use_cache=False)...")

    import json
    from peft import PeftModel

    adapter_cfg     = json.load(open(os.path.join(lora_path, "adapter_config.json")))
    base_model_name = adapter_cfg["base_model_name_or_path"]
    print(f"  base model: {base_model_name}")

    # FastModel로 로드 → apply_qkv 패치 보장
    # use_cache=False → fast_forward_inference RoPE shape 버그 우회
    base_model_inf, tokenizer_inf = FastModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )
    model_inf = PeftModel.from_pretrained(base_model_inf, lora_path)
    model_inf.eval()

    dev_df = load_and_preprocess(os.path.join(DATA_PATH, "dev.csv"), is_train=False)

    preds = []
    for _, row in tqdm(dev_df.iterrows(), total=len(dev_df), desc="dev inference"):
        messages = [
            {"role": "system", "content": QA_SYSTEM},
            {"role": "user",   "content": QA_USER.format(dialogue=row["dialogue"])},
        ]
        text = tokenizer_inf.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
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
                use_cache=False,
                pad_token_id=tokenizer_inf.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        summary   = tokenizer_inf.decode(generated, skip_special_tokens=True).strip()
        preds.append(postprocess(summary))

    os.makedirs(os.path.join(SCRIPT_DIR, "prediction"), exist_ok=True)
    dev_df.assign(pred_summary=preds).to_csv(pred_path, index=False)

    # ── 7. ROUGE 평가 ─────────────────────────────────────────────────────────
    golds  = dev_df["summary"].tolist()
    scores = compute_rouge_combined(preds, golds)

    print(f"\n{'='*60}")
    print(f"[{exp_name}] Dev ROUGE ({scores['method']} 기반)")
    print(f"  R1={scores['r1']:.4f}  R2={scores['r2']:.4f}  RL={scores['rl']:.4f}  Combined={scores['combined']:.4f}")
    print(f"{'='*60}\n")

    del model_inf, base_model_inf
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "exp_name":   exp_name,
        "lora_r":     LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lr":         LR,
        "train_sec":  train_time,
        **scores,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────
def main():
    results = []

    for exp in EXPERIMENTS:
        result = run_experiment(exp)
        results.append(result)

        print("\n[중간 결과 요약]")
        df = pd.DataFrame(results)
        print(df[["exp_name", "lora_r", "lora_alpha", "lr",
                   "r1", "r2", "rl", "combined"]].to_string(index=False))

    df = pd.DataFrame(results)
    df_sorted = df.sort_values("combined", ascending=False).reset_index(drop=True)

    print(f"\n{'='*80}")
    print("최종 결과 요약 (Combined ROUGE 내림차순)")
    print(f"{'='*80}")
    print(df_sorted[["exp_name", "lora_r", "lora_alpha", "lr",
                      "r1", "r2", "rl", "combined"]].to_string(index=False))

    best = df_sorted.iloc[0]
    print(f"\n최고 조합: {best['exp_name']}  Combined={best['combined']:.4f}")
    print("→ 이 설정으로 run_test_inference_q35.py 실행 권장")

    result_csv = os.path.join(SCRIPT_DIR, "prediction/qlora_q35_sweep_results.csv")
    df_sorted.to_csv(result_csv, index=False)
    print(f"\n결과 저장: {result_csv}")


if __name__ == "__main__":
    main()
