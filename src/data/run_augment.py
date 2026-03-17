"""
데이터 증강 실행 스크립트.

augment.py의 BackTranslationAugmenter / EdaAugmenter를 선택 또는 전체 적용해
증강 데이터를 생성하고, 원본 train.csv와 합산한 뒤 output_dir에 저장합니다.

실행 예시:
    python src/data/run_augment.py --method eda
    python src/data/run_augment.py --method back_translation
    python src/data/run_augment.py --method all
    python src/data/run_augment.py --method all --max_samples 1000 --rouge_threshold 0.4
    python src/data/run_augment.py --method eda --data_dir data --output_dir data_aug
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.augment import augment_dataset
from src.data.preprocess import clean_text, filter_by_length


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="데이터 증강 스크립트")
    parser.add_argument(
        "--method",
        choices=["eda", "back_translation", "all"],
        default="eda",
        help="증강 방법 (기본: eda)",
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="원본 CSV 디렉토리 (기본: data)",
    )
    parser.add_argument(
        "--output_dir",
        default="data_aug",
        help="증강 데이터 저장 디렉토리 (기본: data_aug)",
    )
    parser.add_argument(
        "--rouge_threshold",
        type=float,
        default=0.3,
        help="원본과의 ROUGE-L 최솟값 — 이 값 미만은 제거 (기본: 0.3)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="증강할 최대 샘플 수 (기본: 전체)",
    )
    return parser.parse_args()


def run_augmentation(
    train_df: pd.DataFrame,
    method: str,
    rouge_threshold: float,
    max_samples: int | None,
    output_dir: str,
) -> list[pd.DataFrame]:
    """선택된 method에 따라 증강을 실행하고 결과 DataFrame 리스트를 반환합니다."""
    methods = ["eda", "back_translation"] if method == "all" else [method]
    aug_frames: list[pd.DataFrame] = []

    for m in methods:
        out_path = os.path.join(output_dir, f"train_aug_{m}.csv")
        print(f"\n{'='*60}")
        print(f"[Augment] method={m}  threshold={rouge_threshold}  max_samples={max_samples}")
        print(f"{'='*60}")
        aug_df = augment_dataset(
            train_df,
            method=m,
            rouge_threshold=rouge_threshold,
            max_samples=max_samples,
            output_path=out_path,
        )
        aug_frames.append(aug_df)

    return aug_frames


def main() -> None:
    args = parse_args()

    data_dir = os.path.join(_PROJECT_ROOT, args.data_dir)
    output_dir = os.path.join(_PROJECT_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.csv를 찾을 수 없습니다: {train_path}")

    train_df = pd.read_csv(train_path)
    print(f"[Data] 원본 train.csv: {len(train_df)}건")

    # 증강 전 전처리: clean_text → filter_by_length (train.py와 동일 기준)
    train_df = train_df.copy()
    train_df["dialogue"] = train_df["dialogue"].apply(clean_text)
    if "summary" in train_df.columns:
        train_df["summary"] = train_df["summary"].apply(clean_text)
    print("[Preprocess] clean_text 적용 완료")
    train_df = filter_by_length(train_df)
    print(f"[Preprocess] filter_by_length 적용 후: {len(train_df)}건")

    # 증강 실행
    aug_frames = run_augmentation(
        train_df,
        method=args.method,
        rouge_threshold=args.rouge_threshold,
        max_samples=args.max_samples,
        output_dir=output_dir,
    )

    # 원본 + 증강 전체 합산 (컬럼 순서는 원본 train.csv 기준으로 고정)
    original_columns = list(train_df.columns)
    combined = pd.concat([train_df, *aug_frames], ignore_index=True)
    combined = combined[original_columns]  # 원본과 동일한 컬럼 순서·구성 보장
    combined_path = os.path.join(output_dir, "train.csv")
    combined.to_csv(combined_path, index=False)
    aug_total = len(combined) - len(train_df)
    print(f"\n[완료] train.csv: {len(train_df)} + 증강 {aug_total} = {len(combined)}건 → {combined_path}")

    # dev.csv, test.csv 복사 (train.py가 data_dir 단위로 읽기 때문)
    for fname in ("dev.csv", "test.csv"):
        src = os.path.join(data_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"[복사] {fname} → {dst}")
        else:
            print(f"[경고] {fname} 없음: {src}")

    print(f"\n학습 실행:\n  python src/train.py general.data_path={args.output_dir}")


if __name__ == "__main__":
    main()
