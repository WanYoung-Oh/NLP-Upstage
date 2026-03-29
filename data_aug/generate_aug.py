"""
Solar API를 이용한 대화 데이터 증강 스크립트

사용법:
    python data_aug/generate_aug.py \
        --input   data/train.csv \
        --output  data_aug/train_augmented.csv \
        --n_aug   3000 \
        --seed    42 \
        --model   solar-pro \
        --resume
"""

import argparse
import os
import re
import random
import time
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APITimeoutError, APIStatusError
from tqdm import tqdm

from prompt_templates import SYSTEM_PROMPT, make_front_prompt, make_back_prompt

# ── 상수 ──────────────────────────────────────────────────────
ENV_PATH = "/data/ephemeral/home/NLP/.env"
CHECKPOINT_SUFFIX = ".ckpt.csv"
MIN_TURNS = 4          # 이 미만 샘플은 제외 (절반 삭제 시 최소 2턴 보존)
MAX_RETRIES = 3        # API 재시도 횟수
SAVE_EVERY = 100       # N개마다 체크포인트 저장
TURN_PATTERN = re.compile(r"(#Person\d+#:.*?)(?=\n#Person\d+#:|\Z)", re.DOTALL)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── 유틸리티 ──────────────────────────────────────────────────

def parse_turns(dialogue: str) -> list[str]:
    """#PersonN#: 단위로 대화 파싱"""
    turns = TURN_PATTERN.findall(dialogue)
    return [t.strip() for t in turns if t.strip()]


def turns_to_dialogue(turns: list[str]) -> str:
    return "\n".join(turns)


def validate_generated(text: str) -> bool:
    """생성 결과에 #PersonN#: 패턴이 2개 이상 있는지 확인"""
    return len(re.findall(r"#Person\d+#:", text)) >= 2


def extract_turns_from_generated(text: str) -> list[str]:
    """생성 텍스트에서 턴 목록 추출"""
    return [t.strip() for t in TURN_PATTERN.findall(text) if t.strip()]


# ── Solar API 호출 ─────────────────────────────────────────────

def call_solar(
    client: OpenAI,
    model: str,
    user_prompt: str,
) -> str | None:
    """Solar API 호출 (재시도 + 지수 백오프)"""
    wait = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.8,
                max_tokens=1024,
                timeout=60,
            )
            return response.choices[0].message.content
        except RateLimitError:
            log.warning("Rate limit hit. %.1fs 후 재시도 (%d/%d)", wait, attempt, MAX_RETRIES)
            time.sleep(wait)
            wait = min(wait * 2, 30.0)
        except APITimeoutError:
            log.warning("Timeout. %.1fs 후 재시도 (%d/%d)", wait, attempt, MAX_RETRIES)
            time.sleep(wait)
            wait = min(wait * 2, 30.0)
        except APIStatusError as e:
            log.warning("API 오류 %s. %.1fs 후 재시도 (%d/%d)", e.status_code, wait, attempt, MAX_RETRIES)
            time.sleep(wait)
            wait = min(wait * 2, 30.0)
    return None


# ── 단일 샘플 증강 ─────────────────────────────────────────────

def augment_one(
    row: pd.Series,
    client: OpenAI,
    model: str,
    rng: random.Random,
) -> dict | None:
    """
    한 행을 증강:
    - 앞 50% 삭제 → 앞부분 생성 후 [생성 앞 + 기존 뒤] 조합
    - 뒤 50% 삭제 → 뒷부분 생성 후 [기존 앞 + 생성 뒤] 조합
    실패 시 None 반환
    """
    turns = parse_turns(row["dialogue"])
    n = len(turns)
    if n < MIN_TURNS:
        return None

    n_keep = n // 2
    n_gen  = n - n_keep
    delete_front = rng.random() < 0.5  # True = 앞부분 삭제

    if delete_front:
        existing = turns[n_keep:]           # 뒤쪽 보존
        user_prompt = make_front_prompt(
            topic=row["topic"],
            summary=row["summary"],
            existing_dialogue=turns_to_dialogue(existing),
            n_gen=n_gen,
        )
    else:
        existing = turns[:n_keep]           # 앞쪽 보존
        user_prompt = make_back_prompt(
            topic=row["topic"],
            summary=row["summary"],
            existing_dialogue=turns_to_dialogue(existing),
            n_gen=n_gen,
        )

    generated_text = call_solar(client, model, user_prompt)
    if generated_text is None:
        log.warning("API 실패: %s → 스킵", row["fname"])
        return None

    if not validate_generated(generated_text):
        log.warning("형식 불일치: %s → 스킵 (생성: %r...)", row["fname"], generated_text[:80])
        return None

    gen_turns = extract_turns_from_generated(generated_text)

    if delete_front:
        final_turns = gen_turns + existing   # 생성 앞 + 기존 뒤
    else:
        final_turns = existing + gen_turns   # 기존 앞 + 생성 뒤

    return {
        "dialogue": turns_to_dialogue(final_turns),
        "summary":  row["summary"],
        "topic":    row["topic"],
    }


# ── 메인 ──────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # API 키 로드
    load_dotenv(ENV_PATH)
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise RuntimeError(f"UPSTAGE_API_KEY를 {ENV_PATH}에서 찾을 수 없습니다.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.upstage.ai/v1/solar",
    )

    # 원본 데이터 로드 및 필터링
    df = pd.read_csv(args.input)
    df["_turns"] = df["dialogue"].apply(lambda d: len(parse_turns(d)))
    df = df[df["_turns"] >= MIN_TURNS].reset_index(drop=True)
    log.info("필터링 후 샘플 수: %d (원본: 12,457)", len(df))

    # 샘플링
    rng = random.Random(args.seed)
    n_target = min(args.n_aug, len(df))
    indices = list(range(len(df)))
    rng.shuffle(indices)
    selected = df.iloc[indices[:n_target]].reset_index(drop=True)
    log.info("증강 대상: %d개 선택 (seed=%d)", n_target, args.seed)

    # 체크포인트 로드 (--resume)
    output_path = Path(args.output)
    ckpt_path = output_path.parent / (output_path.stem + CHECKPOINT_SUFFIX)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    start_idx = 0

    if args.resume and ckpt_path.exists():
        ckpt_df = pd.read_csv(ckpt_path)
        results = ckpt_df.to_dict("records")
        start_idx = len(results)
        log.info("체크포인트 재개: %d개 이미 완료, %d부터 시작", start_idx, start_idx)

    # 증강 루프
    skipped = 0
    for i in tqdm(range(start_idx, n_target), desc="augment", initial=start_idx, total=n_target):
        row = selected.iloc[i]
        result = augment_one(row, client, args.model, rng)

        if result is None:
            skipped += 1
            continue

        result["fname"] = f"aug_{i:04d}"
        results.append(result)

        # 체크포인트 저장
        if len(results) % SAVE_EVERY == 0:
            _save(results, ckpt_path)
            log.info("체크포인트 저장: %d개 완료 (스킵: %d)", len(results), skipped)

    # 최종 저장
    final_df = pd.DataFrame(results, columns=["fname", "dialogue", "summary", "topic"])
    final_df.to_csv(output_path, index=False)
    log.info("완료: %d개 저장 → %s (스킵: %d)", len(final_df), output_path, skipped)

    # 체크포인트 정리
    if ckpt_path.exists():
        ckpt_path.unlink()


def _save(records: list[dict], path: Path) -> None:
    pd.DataFrame(records).to_csv(path, index=False)


# ── CLI ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Solar API 대화 데이터 증강")
    p.add_argument("--input",   default="/data/ephemeral/home/NLP/data/train.csv",
                   help="원본 train.csv 경로")
    p.add_argument("--output",  default="/data/ephemeral/home/NLP/data_aug/train_augmented.csv",
                   help="결과 저장 경로")
    p.add_argument("--n_aug",   type=int, default=3000, help="생성할 증강 샘플 수")
    p.add_argument("--seed",    type=int, default=42,   help="랜덤 시드")
    p.add_argument("--model",   default="solar-pro",    help="Solar 모델명")
    p.add_argument("--resume",  action="store_true",    help="체크포인트에서 재개")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
