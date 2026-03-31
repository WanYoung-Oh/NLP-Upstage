"""
r4b_response_only_ckpt + proper_noun 포함 전체 프롬프트 MBR 실행 스크립트

- 어댑터: response_only_SFT/r4b_response_only_ckpt
- 프롬프트: mbr_prompts.py의 전체 PROMPT_VARIANTS (proper_noun 포함, 8종)
- 각 variant 완료 시 prediction/ 디렉터리에 CSV 저장
- 전체 완료 후 MBR → 최종 CSV 저장

실행:
    python LLM/run_prompts_proper_noun.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import pandas as pd

SCRIPT_DIR      = Path(__file__).resolve().parent
_DEFAULT_ADAPTER  = SCRIPT_DIR / "response_only_SFT/r4b_response_only_ckpt"
_DEFAULT_TEST_CSV = SCRIPT_DIR / "response_only_SFT/data/test.csv"
_DEFAULT_PRED_DIR = Path("/data/ephemeral/home/NLP/prediction")
LOG_DIR           = SCRIPT_DIR / "logs"

MAX_SEQ_LENGTH = 2048


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_model_and_tokenizer(adapter_path: Path):
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
        attn_implementation="sdpa",  # flex_attention CUDA OOB 방지
    )

    log(f"LoRA 어댑터 로드: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    # tokenizer의 model_max_length(40960)가 generation_config로 전파되어
    # flex_attention 블록마스크 생성 시 RoPE 캐시 초과 문제 방지
    model.generation_config.max_length = MAX_SEQ_LENGTH

    log("토크나이저 로드 (어댑터 폴더 — 학습 시 저장본)")
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path),
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def main(args: argparse.Namespace) -> None:
    adapter_path = Path(args.adapter).resolve()
    test_file    = Path(args.test_csv).resolve()
    pred_dir     = Path(args.pred_dir).resolve()

    pred_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log(f"실행 시작 (timestamp={timestamp})")
    log(f"어댑터: {adapter_path}")
    log(f"테스트: {test_file}")
    log(f"출력 디렉터리: {pred_dir}")

    test_df = pd.read_csv(test_file)
    log(f"테스트 데이터: {len(test_df)}행")

    model, tokenizer = load_model_and_tokenizer(adapter_path)
    log("모델 로드 완료")

    sys.path.insert(0, str(SCRIPT_DIR))
    from prompts.inference import InferencePipeline
    from prompts.mbr_decoding import apply_mbr_to_dataset
    from prompts.mbr_prompts import get_all_prompt_variants

    prompt_variants = list(get_all_prompt_variants().keys())
    log(f"프롬프트 변형 ({len(prompt_variants)}종): {prompt_variants}")

    pipeline = InferencePipeline(model, tokenizer)

    all_predictions: dict[str, list[str]] = {}

    resume_dir = pred_dir / f"proper_noun_variants_{timestamp}"
    if args.resume:
        existing = sorted(pred_dir.glob("proper_noun_variants_*"), reverse=True)
        if existing:
            resume_dir = existing[0]
            log(f"--resume: 기존 variants 폴더 재사용 → {resume_dir}")

    resume_dir.mkdir(parents=True, exist_ok=True)

    for variant in prompt_variants:
        variant_cache = resume_dir / f"{variant}.csv"

        if args.resume and variant_cache.exists():
            cached = pd.read_csv(variant_cache)
            all_predictions[variant] = cached["summary"].tolist()
            log(f"[SKIP] {variant}: 캐시 로드 ({len(all_predictions[variant])}개)")
            continue

        log(f"{'='*60}")
        log(f"추론 시작: {variant}")
        t0 = time.time()

        predictions = pipeline.generate_with_prompts(
            df=test_df,
            prompt_variants=[variant],
            use_topic=(variant == "topic"),
            max_new_tokens=128,
            variants_output_dir=None,
            verbose=True,
        )[variant]

        elapsed = int(time.time() - t0)
        log(f"추론 완료: {variant} | {elapsed//60}m {elapsed%60}s")

        all_predictions[variant] = predictions

        # variant 캐시 저장
        cache_df = pd.DataFrame({"fname": test_df["fname"], "summary": predictions})
        cache_df.to_csv(variant_cache, index=False)

        # 개별 서브미션 CSV 저장
        sub_path = pred_dir / f"proper_noun_{variant}_{timestamp}.csv"
        sub_df = test_df[["fname"]].copy()
        sub_df["summary"] = predictions
        sub_df.to_csv(sub_path, index=False)
        log(f"서브미션 저장: {sub_path}")

    # MBR 앙상블
    log(f"{'='*60}")
    log(f"MBR 앙상블 시작 ({len(prompt_variants)}개 프롬프트)")
    t0 = time.time()

    final_predictions = apply_mbr_to_dataset(
        test_df=test_df,
        all_predictions=all_predictions,
        use_mecab=True,
        verbose=True,
    )

    elapsed = int(time.time() - t0)
    log(f"MBR 완료 | {elapsed//60}m {elapsed%60}s")

    mbr_path = pred_dir / f"proper_noun_mbr_{timestamp}.csv"
    mbr_df = test_df[["fname"]].copy()
    mbr_df["summary"] = final_predictions
    mbr_df.to_csv(mbr_path, index=False)
    log(f"MBR 서브미션 저장: {mbr_path}")

    log(f"{'='*60}")
    log("전체 완료. 생성된 파일:")
    for variant in prompt_variants:
        log(f"  {pred_dir}/proper_noun_{variant}_{timestamp}.csv")
    log(f"  {mbr_path}  ← 최종 MBR 서브미션")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="r4b_response_only_ckpt + proper_noun 포함 8종 프롬프트 MBR"
    )
    p.add_argument("--adapter",   type=str, default=str(_DEFAULT_ADAPTER))
    p.add_argument("--test-csv",  type=str, default=str(_DEFAULT_TEST_CSV))
    p.add_argument("--pred-dir",  type=str, default=str(_DEFAULT_PRED_DIR))
    p.add_argument("--resume",    action="store_true")
    main(p.parse_args())
