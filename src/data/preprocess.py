"""
데이터 전처리 모듈.

- Preprocess: CSV → encoder/decoder 입력 변환
- DatasetForSeq2Seq: 학습/검증용 PyTorch Dataset (train/val 공용)
- DatasetForInference: 추론용 PyTorch Dataset
- clean_text: Phase 3 텍스트 클리닝 (기본적으로 비활성화)
"""

from __future__ import annotations

import os
import re

import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Phase 3: 텍스트 클리닝
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    단독 자음/모음, 불필요한 괄호, 반복 특수기호를 제거합니다.

    Phase 3에서 Preprocess.make_set_as_df() 호출 이후 적용.
    """
    # 단독 자음/모음 연속 제거 (가-힣 완성형 음절·알파벳·숫자·# 에 인접하지 않은 것)
    # \w 는 한글 자모도 포함하므로 [가-힣A-Za-z0-9] 로 명시
    text = re.sub(r"(?<![#가-힣A-Za-z0-9])[ㄱ-ㅎㅏ-ㅣ]+(?![#가-힣A-Za-z0-9])", " ", text)
    # 빈 괄호
    text = re.sub(r"\(\s*\)|\[\s*\]|\{\s*\}", "", text)
    # 반복 특수기호 (3회 이상)
    text = re.sub(r"([^\w\s#])\1{2,}", r"\1", text)
    # 다중 공백 정리
    text = re.sub(r"\s+", " ", text).strip()
    return text


def reverse_utterances(dialogue: str) -> str:
    """
    Phase 5 TTA: 대화 발화 순서를 역전합니다.

    '#Person1#: ...\n#Person2#: ...' 형식의 멀티턴 대화에서
    발화 순서를 뒤집어 TTA 입력으로 사용합니다.
    """
    lines = [l.strip() for l in dialogue.split("\n") if l.strip()]
    reversed_lines = list(reversed(lines))
    return "\n".join(reversed_lines)


def apply_tta(dialogues: list[str], n_ways: int = 2) -> list[list[str]]:
    """
    N-way TTA 적용.

    Args:
        dialogues: 원본 대화 리스트
        n_ways: TTA 변형 수 (1=원본, 2=원본+역전, ...)

    Returns:
        각 대화마다 n_ways 개의 변형 리스트 [[orig, reversed, ...], ...]
    """
    results = []
    for d in dialogues:
        variants = [d]
        if n_ways >= 2:
            variants.append(reverse_utterances(d))
        results.append(variants[:n_ways])
    return results


def filter_by_length(
    df: pd.DataFrame,
    dialogue_max: int = 1500,
    summary_min: int = 50,
    summary_max: int = 250,
) -> pd.DataFrame:
    """Phase 3: 길이 기반 이상치 필터링."""
    before = len(df)
    df = df[df["dialogue"].str.len() <= dialogue_max]
    if "summary" in df.columns:
        df = df[df["summary"].str.len().between(summary_min, summary_max)]
    after = len(df)
    print(f"[Filter] {before} → {after} rows (dropped {before - after})")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Preprocess 클래스
# ---------------------------------------------------------------------------

class Preprocess:
    def __init__(self, bos_token: str, eos_token: str) -> None:
        self.bos_token = bos_token
        self.eos_token = eos_token

    @staticmethod
    def make_set_as_df(file_path: str, is_train: bool = True) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        if is_train:
            return df[["fname", "dialogue", "summary"]]
        else:
            return df[["fname", "dialogue"]]

    def make_input(
        self,
        dataset: pd.DataFrame,
        is_test: bool = False,
        prefix: str = "",
    ) -> tuple:
        """
        encoder/decoder 입력 생성.

        Args:
            dataset: make_set_as_df() 반환값
            is_test: True이면 라벨 없이 encoder 입력만 생성
            prefix: T5 계열 모델용 prefix (예: "summarize: ")

        Returns:
            is_test=False → (encoder_input, decoder_input, decoder_output)
            is_test=True  → (encoder_input, decoder_input)
        """
        dialogues = dataset["dialogue"].apply(lambda x: prefix + str(x))
        if is_test:
            encoder_input = dialogues
            decoder_input = [self.bos_token] * len(dialogues)
            return encoder_input.tolist(), list(decoder_input)
        else:
            encoder_input = dialogues
            decoder_input = dataset["summary"].apply(
                lambda x: self.bos_token + str(x)
            )
            decoder_output = dataset["summary"].apply(
                lambda x: str(x) + self.eos_token
            )
            return (
                encoder_input.tolist(),
                decoder_input.tolist(),
                decoder_output.tolist(),
            )


# ---------------------------------------------------------------------------
# Dataset 클래스
# ---------------------------------------------------------------------------

class DatasetForSeq2Seq(Dataset):
    """학습/검증 공용 Seq2Seq Dataset.

    encoder_input, decoder_input, labels 모두 tokenizer 반환 BatchEncoding.
    __len__은 실제 tensor 크기에서 직접 계산하므로 length 인자와 불일치할 일이 없습니다.
    """

    def __init__(self, encoder_input, decoder_input, labels):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx].clone().detach() for k, v in self.encoder_input.items()}
        item2 = {k: v[idx].clone().detach() for k, v in self.decoder_input.items()}
        item2["decoder_input_ids"] = item2.pop("input_ids")
        item2["decoder_attention_mask"] = item2.pop("attention_mask")
        item.update(item2)
        item["labels"] = self.labels["input_ids"][idx]
        return item

    def __len__(self) -> int:
        return len(self.encoder_input["input_ids"])


# 이전 이름과의 호환성 유지
DatasetForTrain = DatasetForSeq2Seq
DatasetForVal = DatasetForSeq2Seq


class DatasetForInference(Dataset):
    def __init__(self, encoder_input, test_id: pd.Series):
        self.encoder_input = encoder_input
        self.test_id = test_id

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx].clone().detach() for k, v in self.encoder_input.items()}
        item["ID"] = self.test_id.iloc[idx]
        return item

    def __len__(self) -> int:
        return len(self.encoder_input["input_ids"])
