"""
qwen35_9b_lora_sft 전용 프롬프트 앙상블 실행 스크립트

- 모델을 한 번만 로드하여 4개 프롬프트(base, topic, qa_style, gold_mimic) 순차 추론
- 각 프롬프트 완료 시마다 서브미션 CSV 저장
- 전체 완료 후 4-프롬프트 MBR 실행 → 최종 서브미션 CSV 저장
- 토크나이저: 학습 시 어댑터 폴더에 저장된 것 그대로 사용

실행:
    python run_prompts_qwen35.py [--resume]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import pandas as pd

# ── 경로 설정 ─────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).resolve().parent
ADAPTER_PATH  = SCRIPT_DIR / "response_only_SFT/outputs/qwen35_9b_lora_sft/lora_adapter"
TEST_FILE     = SCRIPT_DIR / "response_only_SFT/data/test.csv"
PRED_DIR      = Path("/data/ephemeral/home/prediction")
LOG_DIR       = SCRIPT_DIR / "logs"

PROMPT_VARIANTS = ["base", "topic", "qa_style", "gold_mimic"]
MAX_SEQ_LENGTH  = 2048

# ── 헬퍼 ──────────────────────────────────────────────────────

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_model_and_tokenizer(adapter_path: Path):
    """
    adapter_config.json 기반으로 베이스 모델을 Unsloth FastModel(4-bit)로 로드,
    PeftModel을 얹고 어댑터 폴더의 토크나이저를 사용합니다.
    (학습 시 save_pretrained된 토크나이저를 그대로 사용 — 전제 조건 충족)
    """
    import unsloth  # noqa: F401
    from unsloth import FastModel
    from transformers import AutoTokenizer
    from peft import PeftModel

    adapter_path = adapter_path.resolve()
    cfg_path = adapter_path / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"adapter_config.json 없음: {cfg_path}")

    with open(cfg_path, encoding="utf-8") as f:
        base_model_name = json.load(f)["base_model_name_or_path"]

    log(f"베이스 모델 로드 (Unsloth 4-bit): {base_model_name}")
    base_model, _ = FastModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )

    log(f"LoRA 어댑터 로드: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    log("토크나이저 로드 (어댑터 폴더 — 학습 시 저장본)")
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path),
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def output_csv_path(variant: str, timestamp: str) -> Path:
    return PRED_DIR / f"qwen35_{variant}_{timestamp}.csv"


def mbr_csv_path(timestamp: str) -> Path:
    return PRED_DIR / f"qwen35_mbr_{timestamp}.csv"


# ── 메인 ──────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log(f"실행 시작 (timestamp={timestamp})")
    log(f"어댑터: {ADAPTER_PATH}")
    log(f"테스트: {TEST_FILE}")
    log(f"프롬프트: {PROMPT_VARIANTS}")

    # 테스트 데이터 로드
    test_df = pd.read_csv(TEST_FILE)
    log(f"테스트 데이터: {len(test_df)}행")

    # 모델 로드 (1회)
    model, tokenizer = load_model_and_tokenizer(ADAPTER_PATH)
    log("모델 로드 완료")

    # sys.path에 프로젝트 루트 추가 (prompts 패키지 임포트용)
    sys.path.insert(0, str(SCRIPT_DIR))
    from prompts.inference import InferencePipeline
    from prompts.mbr_decoding import apply_mbr_to_dataset

    pipeline = InferencePipeline(model, tokenizer)

    # ── 프롬프트별 순차 추론 ──────────────────────────────────
    all_predictions: dict[str, list[str]] = {}

    # --resume: 이미 생성된 variant CSV가 있으면 재사용
    resume_dir = PRED_DIR / f"qwen35_variants_{timestamp}"
    if args.resume:
        # 가장 최근 variants 폴더 탐색
        existing = sorted(PRED_DIR.glob("qwen35_variants_*"), reverse=True)
        if existing:
            resume_dir = existing[0]
            log(f"--resume: 기존 variants 폴더 재사용 → {resume_dir}")

    resume_dir.mkdir(parents=True, exist_ok=True)

    for variant in PROMPT_VARIANTS:
        variant_cache = resume_dir / f"{variant}.csv"

        if args.resume and variant_cache.exists():
            cached = pd.read_csv(variant_cache)
            all_predictions[variant] = cached["summary"].tolist()
            log(f"[SKIP] {variant}: 캐시 로드 ({len(all_predictions[variant])}개)")
            continue

        log(f"{'='*60}")
        log(f"추론 시작: {variant}")
        t0 = time.time()

        use_topic = True   # topic 프롬프트는 topic 컬럼 사용; 나머지는 무해
        predictions = pipeline.generate_with_prompts(
            df=test_df,
            prompt_variants=[variant],
            use_topic=use_topic,
            max_new_tokens=128,
            variants_output_dir=None,   # 아래에서 직접 저장
            verbose=True,
        )[variant]

        elapsed = int(time.time() - t0)
        log(f"추론 완료: {variant} | {elapsed//60}m {elapsed%60}s")

        all_predictions[variant] = predictions

        # 서브미션 CSV 저장 (variants 캐시용)
        cache_df = pd.DataFrame({"fname": test_df["fname"], "summary": predictions})
        cache_df.to_csv(variant_cache, index=False)

        # 개별 서브미션 CSV 저장 (prediction/)
        sub_path = output_csv_path(variant, timestamp)
        sub_df = test_df[["fname"]].copy()
        sub_df["summary"] = predictions
        sub_df.to_csv(sub_path, index=False)
        log(f"서브미션 저장: {sub_path}")

    # ── MBR 앙상블 ───────────────────────────────────────────
    log(f"{'='*60}")
    log(f"MBR 앙상블 시작 ({len(PROMPT_VARIANTS)}개 프롬프트)")
    t0 = time.time()

    final_predictions = apply_mbr_to_dataset(
        test_df=test_df,
        all_predictions=all_predictions,
        use_mecab=True,
        verbose=True,
    )

    elapsed = int(time.time() - t0)
    log(f"MBR 완료 | {elapsed//60}m {elapsed%60}s")

    mbr_path = mbr_csv_path(timestamp)
    mbr_df = test_df[["fname"]].copy()
    mbr_df["summary"] = final_predictions
    mbr_df.to_csv(mbr_path, index=False)
    log(f"MBR 서브미션 저장: {mbr_path}")

    log(f"{'='*60}")
    log("전체 완료. 생성된 파일:")
    for variant in PROMPT_VARIANTS:
        log(f"  {output_csv_path(variant, timestamp)}")
    log(f"  {mbr_path}  ← 최종 MBR 서브미션")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true",
                   help="기존 variant 결과가 있으면 재사용하여 이어서 실행")
    main(p.parse_args())
