"""
데이터 전처리 모듈.

- Preprocess: CSV → encoder/decoder 입력 변환
- DatasetForSeq2Seq: 학습/검증용 PyTorch Dataset (train/val 공용)
- DatasetForInference: 추론용 PyTorch Dataset
- clean_text: Phase 3 텍스트 클리닝 (기본적으로 비활성화)
"""

from __future__ import annotations

import os
import random
import re

import pandas as pd
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Topic prefix (방법 A, RESEARCH.md §13)
# ---------------------------------------------------------------------------

_TOPIC_MASK = "[MASK]"


def build_topic_prefix(topic: str, mask_prob: float = 0.0) -> str:
    """topic을 '[TOPIC] ... [SEP]\\n' 형식의 encoder prefix로 변환합니다.

    Args:
        topic: 대화 주제 문자열.
        mask_prob: 이 확률로 topic 을 [MASK] 로 대체합니다.
            학습 시 0.25 권장, 검증/추론 시 0.0.

    Returns:
        "[TOPIC] {topic} [SEP]\\n" 또는 "[TOPIC] [MASK] [SEP]\\n"
    """
    if not topic or topic in ("nan", _TOPIC_MASK):
        return f"[TOPIC] {_TOPIC_MASK} [SEP]\n"
    if mask_prob > 0.0 and random.random() < mask_prob:
        return f"[TOPIC] {_TOPIC_MASK} [SEP]\n"
    return f"[TOPIC] {topic} [SEP]\n"


# ---------------------------------------------------------------------------
# Phase 3: 텍스트 클리닝
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    단독 자음/모음, 불필요한 괄호, 반복 특수기호를 제거합니다.

    Phase 3에서 Preprocess.make_set_as_df() 호출 이후 적용.
    """
    # 1. 자음/모음 제거 (단, 해시태그 뒤의 자음은 유지하고 싶다면 (?<!#) 추가)
    # "이거ㅋㅋ"도 지우려면 전후방 탐색을 완전히 제거하거나 조절해야 합니다.
    text = re.sub(r"[ㄱ-ㅎㅏ-ㅣ]+", " ", text)

    # 2. 내용 없는 빈 괄호 제거 ( (), [], {} : 실제 데이터를 확인해 보니 해당 사항은 없는 것으로 확인되나 코드는 유지하기로 결정함)
    text = re.sub(r"\(\s*\)|\[\s*\]|\{\s*\}", "", text)

    # 3. 반복되는 특수기호 축약 (3회 이상 반복 시 1개로 축약, 단 #은 예외 가능)
    # 예: !!! -> !, ??? -> ?
    text = re.sub(r"([^\w\s#])\1{2,}", r"\1", text)

    # 4. 다중 공백 정리 및 양끝 공백 제거
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
    dialogue_max: int = 2300,  # train max 2,168, dev max 1,269, test max 2,275
    summary_min: int = 10,     # train min 13, dev min 29
    summary_max: int = 377,    # train max 376, dev max 283
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
            cols = ["fname", "dialogue", "summary"]
            if "topic" in df.columns:
                cols.append("topic")
            return df[cols]
        else:
            return df[["fname", "dialogue"]]

    def make_input(
        self,
        dataset: pd.DataFrame,
        is_test: bool = False,
        prefix: str = "",
        use_topic: bool = False,
        topic_mask_prob: float = 0.0,
    ) -> tuple:
        """
        encoder/decoder 입력 생성.

        Args:
            dataset: make_set_as_df() 반환값
            is_test: True이면 라벨 없이 encoder 입력만 생성
            prefix: T5 계열 모델용 prefix (예: "summarize: ")
            use_topic: True이면 topic을 encoder 입력 앞에 prepend (방법 A)
            topic_mask_prob: topic을 [MASK]로 대체할 확률 (학습 시 0.25 권장)
                topic 컬럼이 없는 경우(test.csv) 항상 [MASK] 사용

        Returns:
            is_test=False → (encoder_input, decoder_input, decoder_output)
            is_test=True  → (encoder_input, decoder_input)
        """
        if use_topic:
            if "topic" in dataset.columns:
                topics = dataset["topic"].fillna(_TOPIC_MASK).astype(str)
            else:
                topics = pd.Series([_TOPIC_MASK] * len(dataset), index=dataset.index)
            topic_prefixes = topics.apply(
                lambda t: build_topic_prefix(t, mask_prob=topic_mask_prob)
            )
            dialogues = topic_prefixes + dataset["dialogue"].apply(
                lambda x: prefix + str(x)
            )
        else:
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


class DatasetForCausalLM(Dataset):
    """Causal LM (decoder-only) 학습/검증용 Dataset.

    prompt + response를 하나의 시퀀스로 합치고,
    prompt 위치의 labels를 -100으로 마스킹해 loss 계산에서 제외합니다.
    """

    def __init__(
        self,
        input_ids: torch.Tensor,       # (N, seq_len)
        attention_mask: torch.Tensor,  # (N, seq_len)
        labels: torch.Tensor,          # (N, seq_len), -100 at prompt positions
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

    def __len__(self) -> int:
        return len(self.input_ids)


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
