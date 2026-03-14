"""
추론 메인 스크립트.

실행 예시:
    # Seq2Seq beam search (기본)
    python src/inference.py inference.ckt_path=outputs/.../checkpoints/best_model

    # beam8 설정으로 변경
    python src/inference.py inference=beam8 inference.ckt_path=...

    # TTA (발화 역전 N-way + MBR 선택)
    python src/inference.py inference=tta inference.ckt_path=...

    # Solar API
    python src/inference.py inference=solar_api
"""

from __future__ import annotations

import os
import sys
import time

_LAUNCH_DIR = os.getcwd()
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.data.preprocess import DatasetForInference, Preprocess, apply_tta
from src.utils.device import get_device
from src.utils.postprocess import batch_postprocess_with_flags


def _resolve(raw: str) -> str:
    """상대 경로를 _LAUNCH_DIR 기준 절대 경로로 변환합니다."""
    if os.path.isabs(raw):
        return raw
    return os.path.join(_LAUNCH_DIR, raw)


# ---------------------------------------------------------------------------
# Seq2Seq Inferencer (beam search / MBR / TTA)
# ---------------------------------------------------------------------------

class Seq2SeqInferencer:
    """
    Seq2Seq 기반 추론기.

    - n_tta_ways=1 (기본): 일반 beam search / MBR
    - n_tta_ways>=2: TTA — 발화 역전 등 N-way 변형 생성 후 MBR로 최선 선택
    - do_sample=True: MBR decoding (다수 샘플 후 ROUGE-L 최고 선택)
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = get_device()

    def _load_model_and_tokenizer(self):
        cfg = self.cfg
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(cfg.tokenizer.special_tokens)}
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg.inference.ckt_path)
        model.resize_token_embeddings(len(tokenizer))
        model.to(self.device)
        model.eval()
        return model, tokenizer

    def _generate(self, model, tokenizer, batch: dict, effective_max: int) -> list[str]:
        """단일 배치에서 beam search 또는 MBR 샘플링으로 요약문 생성."""
        cfg = self.cfg
        is_mbr = getattr(cfg.inference, "do_sample", False)

        gen_kwargs: dict = dict(
            input_ids=batch["input_ids"].to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            max_new_tokens=effective_max,
        )

        if is_mbr:
            n = getattr(cfg.inference, "n_samples", 10)
            gen_kwargs.update(
                do_sample=True,
                temperature=getattr(cfg.inference, "temperature", 0.9),
                top_p=getattr(cfg.inference, "top_p", 0.95),
                num_return_sequences=n,
            )
            generated = model.generate(**gen_kwargs)
            batch_size = batch["input_ids"].size(0)
            summaries = []
            for i in range(batch_size):
                candidates = [tokenizer.decode(generated[i * n + j]) for j in range(n)]
                summaries.append(_mbr_select(candidates))
        else:
            gen_kwargs.update(
                num_beams=cfg.inference.num_beams,
                no_repeat_ngram_size=cfg.inference.no_repeat_ngram_size,
                early_stopping=cfg.inference.early_stopping,
                length_penalty=cfg.inference.length_penalty,
            )
            generated = model.generate(**gen_kwargs)
            summaries = [tokenizer.decode(ids) for ids in generated]

        return summaries

    def _effective_max(self, batch: dict) -> int:
        """max_length_ratio에 따라 동적 max_new_tokens 계산."""
        cfg = self.cfg
        max_length_ratio: float = getattr(cfg.inference, "max_length_ratio", 0.0)
        if max_length_ratio > 0:
            input_len = int(batch["attention_mask"].sum(dim=1).float().max().item())
            dynamic_max = max(30, int(input_len * max_length_ratio))
            return min(dynamic_max, cfg.inference.generate_max_length)
        return cfg.inference.generate_max_length

    def run(self) -> pd.DataFrame:
        cfg = self.cfg
        n_tta_ways: int = getattr(cfg.inference, "n_tta_ways", 1)

        model, tokenizer = self._load_model_and_tokenizer()

        prefix = getattr(cfg.model, "prefix", "")
        preprocessor = Preprocess(cfg.tokenizer.bos_token, cfg.tokenizer.eos_token)
        test_file = os.path.join(_resolve(cfg.general.data_path), "test.csv")
        test_data = preprocessor.make_set_as_df(test_file, is_train=False)
        enc_input_raw, _ = preprocessor.make_input(test_data, is_test=True, prefix=prefix)

        if n_tta_ways >= 2:
            return self._run_tta(model, tokenizer, enc_input_raw, test_data, n_tta_ways)
        else:
            return self._run_standard(model, tokenizer, enc_input_raw, test_data)

    def _run_standard(self, model, tokenizer, enc_input_raw: list[str], test_data) -> pd.DataFrame:
        """일반 beam search / MBR 추론."""
        cfg = self.cfg
        tok_kw = dict(
            return_tensors="pt", padding=True, add_special_tokens=True,
            truncation=True, max_length=cfg.tokenizer.encoder_max_len,
            return_token_type_ids=False,
        )
        tokenized_enc = tokenizer(enc_input_raw, **tok_kw)
        dataset = DatasetForInference(tokenized_enc, test_data["fname"])
        dataloader = DataLoader(dataset, batch_size=cfg.inference.batch_size)

        summaries: list[str] = []
        text_ids: list[str] = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference"):
                text_ids.extend(batch["ID"])
                effective_max = self._effective_max(batch)
                summaries.extend(self._generate(model, tokenizer, batch, effective_max))

        return self._save(summaries, text_ids)

    def _run_tta(
        self,
        model,
        tokenizer,
        enc_input_raw: list[str],
        test_data,
        n_tta_ways: int,
    ) -> pd.DataFrame:
        """
        TTA 추론: N-way 변형(원본 + 발화 역전 등)으로 각각 요약 생성 후
        MBRDecoder로 최적 후보 선택.
        """
        cfg = self.cfg
        from src.ensemble import MBRDecoder

        # 각 샘플별 N개 변형 생성 [[orig, reversed, ...], ...]
        tta_variants_per_sample: list[list[str]] = apply_tta(enc_input_raw, n_ways=n_tta_ways)

        # Flat 리스트로 펼쳐서 한 번에 토크나이징
        flat_inputs: list[str] = [v for variants in tta_variants_per_sample for v in variants]
        actual_n = len(tta_variants_per_sample[0]) if tta_variants_per_sample else n_tta_ways

        tok_kw = dict(
            return_tensors="pt", padding=True, add_special_tokens=True,
            truncation=True, max_length=cfg.tokenizer.encoder_max_len,
            return_token_type_ids=False,
        )
        tokenized_enc = tokenizer(flat_inputs, **tok_kw)

        # fname을 flat 인덱스용으로 확장 (MBR 그룹핑에는 사용 안 하지만 DatasetForInference 필요)
        flat_fnames = pd.Series([
            test_data["fname"].iloc[i // actual_n]
            for i in range(len(flat_inputs))
        ])
        dataset = DatasetForInference(tokenized_enc, flat_fnames)
        dataloader = DataLoader(dataset, batch_size=cfg.inference.batch_size)

        flat_summaries: list[str] = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"TTA Inference ({actual_n}-way)"):
                effective_max = self._effective_max(batch)
                batch_summaries = self._generate(model, tokenizer, batch, effective_max)
                flat_summaries.extend(batch_summaries)

        # 샘플별 그룹화 후 MBR 선택
        decoder = MBRDecoder()
        num_samples = len(tta_variants_per_sample)
        final_summaries: list[str] = []
        text_ids: list[str] = []

        for i in range(num_samples):
            candidates = flat_summaries[i * actual_n : (i + 1) * actual_n]
            best = decoder.decode(candidates)
            final_summaries.append(best)
            text_ids.append(test_data["fname"].iloc[i])

        return self._save(final_summaries, text_ids)

    def _save(self, summaries: list[str], text_ids: list[str]) -> pd.DataFrame:
        """후처리 + CSV 저장."""
        cfg = self.cfg
        remove_tokens = list(cfg.inference.remove_tokens)
        processed, flags = batch_postprocess_with_flags(summaries, remove_tokens)

        short_count = sum(flags)
        if short_count > 0:
            print(f"[경고] {short_count}/{len(processed)}개 요약문이 최소 길이 미달 (재생성 권장)")

        result_path = _resolve(cfg.inference.result_path)
        os.makedirs(result_path, exist_ok=True)
        output_filename = getattr(cfg.inference, "output_filename", "output.csv")
        output = pd.DataFrame({"fname": text_ids, "summary": processed})
        out_path = os.path.join(result_path, output_filename)
        output.to_csv(out_path, index=False)
        print(f"Saved → {out_path}")
        return output


def _mbr_select(candidates: list[str]) -> str:
    """N개 후보 중 다른 후보들과의 평균 ROUGE-L이 가장 높은 문장 선택."""
    if not candidates:
        return ""

    from rouge import Rouge

    rouge = Rouge()
    best, best_score = candidates[0], -1.0
    for i, cand in enumerate(candidates):
        others = [c for j, c in enumerate(candidates) if j != i and c.strip()]
        if not others:
            continue
        cand_safe = cand.strip() if cand.strip() else "."
        try:
            scores = rouge.get_scores(
                [cand_safe] * len(others), others, avg=True
            )
            avg = scores["rouge-l"]["f"]
        except Exception:
            avg = 0.0
        if avg > best_score:
            best_score = avg
            best = cand
    return best


# ---------------------------------------------------------------------------
# Solar API Inferencer (Phase 4)
# ---------------------------------------------------------------------------

class SolarAPIInferencer:
    """Upstage Solar Chat API를 이용한 few-shot 요약 추론."""

    SYSTEM_PROMPT = (
        "당신은 한국어 대화 요약 전문가입니다. "
        "주어진 대화를 아래 기준에 따라 요약하세요.\n"
        "1. 대화의 가장 중요한 정보를 전달합니다.\n"
        "2. 요약문은 대화 길이의 20% 이내로 간략하게 작성합니다.\n"
        "3. 사람 이름·기업명 등 대화 내 중요한 명명된 개체를 그대로 보존합니다.\n"
        "4. 화자의 의도를 이해하고 관찰자의 관점(3인칭)에서 작성합니다.\n"
        "5. 은어·약어 없이 공식적으로 사용되는 표준어로 작성합니다."
    )

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        try:
            from openai import OpenAI  # type: ignore

            api_key = os.environ.get("UPSTAGE_API_KEY", "")
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.upstage.ai/v1/solar",
            )
        except ImportError as e:
            raise ImportError("openai 설치 필요: pip install openai") from e

        # BM25 인덱스는 run() 진입 시 한 번만 생성합니다.
        self._bm25 = None
        self._train_df: pd.DataFrame | None = None

    def build_prompt(
        self,
        dialogue: str,
        few_shot_examples: list[dict] | None = None,
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        if few_shot_examples:
            for ex in few_shot_examples:
                messages.append({"role": "user", "content": f"대화:\n{ex['dialogue']}"})
                messages.append({"role": "assistant", "content": ex["summary"]})
        messages.append({"role": "user", "content": f"대화:\n{dialogue}"})
        return messages

    def summarize(self, dialogue: str, few_shot: list[dict] | None = None) -> str:
        cfg = self.cfg
        messages = self.build_prompt(dialogue, few_shot)
        response = self.client.chat.completions.create(
            model=cfg.inference.model_name,
            messages=messages,
            temperature=cfg.inference.temperature,
            top_p=cfg.inference.top_p,
            max_tokens=cfg.inference.max_tokens,
        )
        return response.choices[0].message.content.strip()

    def _build_bm25_index(self, train_df: pd.DataFrame) -> None:
        """BM25 인덱스를 한 번만 구축합니다."""
        try:
            from rank_bm25 import BM25Okapi  # type: ignore

            corpus = [d.split() for d in train_df["dialogue"].tolist()]
            self._bm25 = BM25Okapi(corpus)
            self._train_df = train_df
        except ImportError:
            self._bm25 = None

    def _load_few_shot_examples(self, dialogue: str) -> list[dict]:
        """BM25 기반 동적 few-shot 예제 선택 (인덱스는 사전 구축된 것 사용)."""
        cfg = self.cfg
        n = cfg.inference.n_few_shot
        train_df = self._train_df

        if train_df is None:
            return []

        if self._bm25 is not None:
            query = dialogue.split()
            scores = self._bm25.get_scores(query)
            top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
            return train_df.iloc[top_idx][["dialogue", "summary"]].to_dict("records")

        return train_df.head(n)[["dialogue", "summary"]].to_dict("records")

    def run(self) -> pd.DataFrame:
        cfg = self.cfg
        data_path = _resolve(cfg.general.data_path)
        test_data = pd.read_csv(os.path.join(data_path, "test.csv"))

        use_few_shot = getattr(cfg.inference, "prompt_style", "zero_shot") == "few_shot"
        if use_few_shot:
            train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
            self._train_df = train_df
            if getattr(cfg.inference, "use_bm25", False):
                self._build_bm25_index(train_df)

        rate_limit = getattr(cfg.inference, "rate_limit_rpm", 100)
        delay = 60.0 / rate_limit

        summaries, text_ids = [], []
        for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Solar API"):
            few_shot = self._load_few_shot_examples(row["dialogue"]) if use_few_shot else None
            summary = self.summarize(row["dialogue"], few_shot)
            summaries.append(summary)
            text_ids.append(row["fname"])
            time.sleep(delay)

        remove_tokens = list(cfg.inference.remove_tokens)
        processed, flags = batch_postprocess_with_flags(summaries, remove_tokens)
        short_count = sum(flags)
        if short_count > 0:
            print(f"[경고] {short_count}/{len(processed)}개 요약문이 최소 길이 미달 (재생성 권장)")

        result_path = _resolve(cfg.inference.result_path)
        os.makedirs(result_path, exist_ok=True)
        output_filename = getattr(cfg.inference, "output_filename", "output_solar.csv")
        output = pd.DataFrame({"fname": text_ids, "summary": processed})
        out_path = os.path.join(result_path, output_filename)
        output.to_csv(out_path, index=False)
        print(f"Saved → {out_path}")
        return output


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # inference_type: solar_api 플래그로 명시적 분기 (없으면 model_name 폴백)
    inference_type = getattr(cfg.inference, "inference_type", None)
    if inference_type == "solar_api" or (
        inference_type is None
        and hasattr(cfg.inference, "model_name")
        and "solar" in str(getattr(cfg.inference, "model_name", "")).lower()
    ):
        SolarAPIInferencer(cfg).run()
    else:
        Seq2SeqInferencer(cfg).run()


if __name__ == "__main__":
    main()
