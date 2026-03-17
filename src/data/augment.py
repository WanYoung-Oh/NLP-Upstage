"""
Phase 3: 데이터 증강 모듈.

- BackTranslationAugmenter: ko → en → ko 역번역
- EdaAugmenter: nlpaug 기반 EDA/AEDA 증강
- augment_dataset: 증강 데이터 생성 후 ROUGE 필터링
"""

from __future__ import annotations

import os
import re
import time
import pandas as pd
from tqdm import tqdm

# 대화에서 화자 토큰을 분리하는 패턴: #Person1#, #Person2#, ... 등
_SPEAKER_PATTERN = re.compile(r"(#\w+#:?\s*)")


# ---------------------------------------------------------------------------
# Back-Translation
# ---------------------------------------------------------------------------

class BackTranslationAugmenter:
    """deep-translator를 이용한 ko → en → ko 역번역 증강."""

    def __init__(self, src_lang: str = "ko", pivot_lang: str = "en") -> None:
        try:
            from deep_translator import GoogleTranslator  # type: ignore
            self._GoogleTranslator = GoogleTranslator
        except ImportError as e:
            raise ImportError("deep-translator 설치 필요: pip install deep-translator") from e
        self.src = src_lang
        self.pivot = pivot_lang

    def augment(self, text: str, delay: float = 0.3) -> str:
        try:
            # 화자 토큰(#Person1#: 등)을 기준으로 분리하여 발화 내용에만 번역 적용
            parts = _SPEAKER_PATTERN.split(text)
            result_parts: list[str] = []
            for part in parts:
                if _SPEAKER_PATTERN.fullmatch(part) or not part.strip():
                    # 화자 토큰과 빈 문자열은 원본 그대로 유지
                    result_parts.append(part)
                else:
                    # 발화 내용만 ko → en → ko 번역
                    to_pivot = self._GoogleTranslator(source=self.src, target=self.pivot)
                    en = to_pivot.translate(part)
                    time.sleep(delay)
                    to_src = self._GoogleTranslator(source=self.pivot, target=self.src)
                    ko = to_src.translate(en)
                    time.sleep(delay)
                    result_parts.append(ko)
            return "".join(result_parts)
        except Exception:
            return text  # 실패 시 원본 반환


# ---------------------------------------------------------------------------
# EDA / AEDA (nlpaug 기반)
# ---------------------------------------------------------------------------

class EdaAugmenter:
    """nlpaug를 이용한 EDA/AEDA 텍스트 증강."""

    def __init__(self, aug_p: float = 0.05) -> None:
        try:
            import nlpaug.augmenter.word as naw  # type: ignore
        except ImportError as e:
            raise ImportError("nlpaug 설치 필요: pip install nlpaug") from e
        self.aug = naw.RandomWordAug(action="delete", aug_p=aug_p)

    def augment(self, text: str) -> str:
        try:
            # 화자 토큰(#Person1#: 등)을 기준으로 분리
            # split 결과는 [비화자, 화자토큰, 발화내용, 화자토큰, 발화내용, ...] 형태
            parts = _SPEAKER_PATTERN.split(text)
            result_parts: list[str] = []
            for part in parts:
                if _SPEAKER_PATTERN.fullmatch(part):
                    # 화자 토큰은 원본 그대로 유지
                    result_parts.append(part)
                elif part.strip():
                    # 실제 발화 내용에만 EDA 적용
                    aug = self.aug.augment(part)
                    result_parts.append(aug[0] if isinstance(aug, list) else aug)
                else:
                    result_parts.append(part)
            return "".join(result_parts)
        except Exception:
            return text


# ---------------------------------------------------------------------------
# 증강 파이프라인
# ---------------------------------------------------------------------------

def augment_dataset(
    df: pd.DataFrame,
    method: str = "back_translation",
    rouge_threshold: float = 0.3,
    max_samples: int | None = None,
    output_path: str = "data/train_aug.csv",
) -> pd.DataFrame:
    """
    데이터 증강 후 ROUGE 필터링을 거쳐 augmented CSV를 저장합니다.

    Args:
        df: 원본 train DataFrame (fname, dialogue, summary 컬럼 필요)
        method: "back_translation" | "eda"
        rouge_threshold: 원본과의 최소 ROUGE-L 유사도 (너무 낮으면 제거)
        max_samples: 최대 증강 샘플 수 (None이면 전체)
        output_path: 증강 데이터 저장 경로
    """
    from rouge import Rouge

    if method == "back_translation":
        augmenter: BackTranslationAugmenter | EdaAugmenter = BackTranslationAugmenter()
    elif method == "eda":
        augmenter = EdaAugmenter()
    else:
        raise ValueError(f"Unknown method: {method}")

    rouge = Rouge()
    target = df if max_samples is None else df.head(max_samples)
    augmented_rows = []

    for _, row in tqdm(target.iterrows(), total=len(target), desc=f"Augmenting ({method})"):
        aug_dialogue = augmenter.augment(row["dialogue"])

        # ROUGE 필터링: 원본과 증강 dialogue 유사도 검사
        try:
            score = rouge.get_scores(aug_dialogue, row["dialogue"])[0]["rouge-l"]["f"]
        except Exception:
            score = 0.0

        if score >= rouge_threshold:
            aug_row = row.to_dict()
            aug_row["fname"] = str(row["fname"]) + "_aug"
            aug_row["dialogue"] = aug_dialogue
            augmented_rows.append(aug_row)

    aug_df = pd.DataFrame(augmented_rows)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    aug_df.to_csv(output_path, index=False)
    print(f"[Augment] {len(aug_df)} samples saved to {output_path}")
    return aug_df
