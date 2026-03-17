"""
학습 메인 스크립트.

실행 예시:
    python src/train.py                                          # 기본 (kobart + baseline)
    python src/train.py model=kot5 training=full                 # KoT5 풀 학습
    python src/train.py training.learning_rate=5e-5              # LR override
    python src/train.py training.use_all_data=true               # Train+Dev 합산 (최종 제출용)
    python src/train.py -m model=kobart,kot5 training=baseline   # Hydra sweep
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from datetime import datetime

# Hydra가 @hydra.main 진입 시 CWD를 outputs/ 로 변경합니다.
# 모듈 로드 시점(변경 전)의 CWD와 프로젝트 루트를 미리 캡처합니다.
_LAUNCH_DIR = os.getcwd()
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

import hydra
import pandas as pd
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import (
    EarlyStoppingCallback,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from src.data.preprocess import DatasetForCausalLM, DatasetForSeq2Seq, Preprocess, build_topic_prefix, clean_text, filter_by_length, _TOPIC_MASK
from src.models.summarizer import load_tokenizer_and_model
from src.utils.device import get_device
from src.utils.metrics import compute_metrics


def _build_generation_config(cfg: DictConfig, tokenizer=None, model=None) -> GenerationConfig | None:
    """평가 시 predict_with_generate에 쓸 GenerationConfig.

    custom GenerationConfig를 Trainer에 넘기면 모델 기본값을 완전히 override하므로,
    decoder_start_token_id / bos_token_id / eos_token_id를 명시해야 평가 시 오류가 발생하지 않는다.
    (transformers ≥ 4.38 에서 누락 시 ValueError 발생)
    """
    if not getattr(cfg.training, "predict_with_generate", False):
        return None
    max_length = getattr(cfg.training, "generation_max_length", 100)
    num_beams = getattr(cfg.training, "generation_num_beams", 4)
    repetition_penalty = getattr(cfg.training, "generation_repetition_penalty", 2.0)
    no_repeat_ngram_size = getattr(cfg.training, "generation_no_repeat_ngram_size", 3)

    # decoder_start_token_id: 모델 config → tokenizer bos → pad 순으로 fallback
    decoder_start_token_id: int | None = None
    if model is not None:
        decoder_start_token_id = getattr(model.config, "decoder_start_token_id", None)
    if decoder_start_token_id is None and tokenizer is not None:
        decoder_start_token_id = getattr(tokenizer, "bos_token_id", None)
    if decoder_start_token_id is None and tokenizer is not None:
        decoder_start_token_id = getattr(tokenizer, "pad_token_id", None)

    bos_token_id: int | None = getattr(tokenizer, "bos_token_id", None) if tokenizer else None
    eos_token_id: int | None = getattr(tokenizer, "eos_token_id", None) if tokenizer else None

    return GenerationConfig(
        max_length=max_length,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=True,
        decoder_start_token_id=decoder_start_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )


def _next_run_id(checkpoints_root: str) -> str:
    """yymmdd_run_NNN 형식의 다음 run_id를 반환합니다."""
    date_prefix = datetime.now().strftime("%y%m%d")
    pattern = re.compile(rf"^{date_prefix}_run_(\d+)$")
    existing = [
        int(m.group(1))
        for name in (os.listdir(checkpoints_root) if os.path.isdir(checkpoints_root) else [])
        if (m := pattern.match(name))
    ]
    return f"{date_prefix}_run_{(max(existing, default=0) + 1):03d}"


class BestCheckpointCallback(TrainerCallback):
    """체크포인트를 epoch{##}_{score:.4f} 형식으로 저장하고 상위 top_k개만 유지합니다.

    architecture="causal_lm" 인 경우 eval_loss를 1/(1+loss) 로 변환해
    "높을수록 좋음" 형식으로 통일합니다. 이렇게 하면 seq2seq와 동일한
    _find_best_checkpoint 로직으로 앙상블 시 best 체크포인트를 선택할 수 있습니다.
    """

    def __init__(self, output_dir: str, top_k: int = 3, architecture: str = "seq2seq"):
        self.output_dir = output_dir
        self.top_k = top_k
        self.architecture = architecture
        self._last_score: float = 0.0
        self._last_epoch: int = 0
        self._checkpoints: list[tuple[float, str]] = []

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics:
            if self.architecture == "causal_lm":
                # eval_loss는 낮을수록 좋으므로 1/(1+loss) 변환으로 "높을수록 좋음" 형식 통일
                loss = metrics.get("eval_loss", 0.0)
                self._last_score = 1.0 / (1.0 + loss)
            else:
                self._last_score = metrics.get("eval_rouge_combined", 0.0)
            self._last_epoch = round(state.epoch) if state.epoch else 0

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # rename 전 경로를 변수에 캡처해두어 state 비교에 사용합니다.
        src = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(src):
            return
        dst_name = f"epoch{self._last_epoch:02d}_{self._last_score:.4f}"
        dst = os.path.join(self.output_dir, dst_name)
        os.rename(src, dst)
        # Trainer는 state.best_model_checkpoint를 rename 전 경로(src)로 기록합니다.
        # rename 후 경로(dst)로 갱신해 EarlyStoppingCallback 등이 올바른 경로를 참조하도록 합니다.
        if state.best_model_checkpoint == src:
            state.best_model_checkpoint = dst
        self._checkpoints.append((self._last_score, dst))
        self._checkpoints.sort(key=lambda x: x[0], reverse=True)
        while len(self._checkpoints) > self.top_k:
            _, to_remove = self._checkpoints.pop()
            if os.path.isdir(to_remove):
                shutil.rmtree(to_remove)


def _resolve_data_path(raw: str) -> str:
    """상대 경로를 실행 시점 디렉토리(_LAUNCH_DIR) 기준으로 절대 경로로 변환합니다."""
    if os.path.isabs(raw):
        return raw
    return os.path.join(_LAUNCH_DIR, raw)


def _build_causal_lm_dataset(
    df: pd.DataFrame,
    tokenizer,
    max_length: int,
    use_topic: bool = False,
    topic_mask_prob: float = 0.0,
) -> DatasetForCausalLM:
    """Causal LM용 Dataset 생성.

    각 샘플을 "[INST] {topic_line}다음 대화를 한국어로 요약하세요:\n{dialogue}\n[/INST]\n{summary}</s>"
    형태로 구성하고 prompt 위치의 labels는 -100으로 마스킹합니다.

    use_topic=True이면 프롬프트에 "주제: {topic}\n" 라인이 추가됩니다.
    topic 컬럼이 없거나 topic_mask_prob 확률에 걸리면 "주제: [MASK]\n"로 대체됩니다.
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    all_input_ids: list[list[int]] = []
    all_attention_mask: list[list[int]] = []
    all_labels: list[list[int]] = []

    for _, row in df.iterrows():
        dialogue = str(row["dialogue"])
        summary = str(row.get("summary", ""))

        if use_topic:
            raw_topic = str(row["topic"]) if "topic" in row.index else _TOPIC_MASK
            if not raw_topic or raw_topic == "nan":
                raw_topic = _TOPIC_MASK
            # build_topic_prefix 대신 causal LM 형식("주제: ...")을 직접 구성
            import random as _random
            if raw_topic == _TOPIC_MASK or (topic_mask_prob > 0.0 and _random.random() < topic_mask_prob):
                topic_line = f"주제: {_TOPIC_MASK}\n"
            else:
                topic_line = f"주제: {raw_topic}\n"
        else:
            topic_line = ""

        prompt = f"[INST] {topic_line}다음 대화를 한국어로 요약하세요:\n{dialogue}\n[/INST]\n"
        response = summary + tokenizer.eos_token

        prompt_ids: list[int] = tokenizer.encode(prompt, add_special_tokens=True)
        response_ids: list[int] = tokenizer.encode(response, add_special_tokens=False)

        full_ids = prompt_ids + response_ids
        n_prompt = len(prompt_ids)

        if len(full_ids) > max_length:
            max_prompt_len = max_length - len(response_ids)
            if max_prompt_len <= 0:
                full_ids = full_ids[:max_length]
                n_prompt = 0
            else:
                prompt_ids = prompt_ids[:max_prompt_len]
                n_prompt = len(prompt_ids)
                full_ids = prompt_ids + response_ids

        seq_len = len(full_ids)
        pad_len = max_length - seq_len

        labels = [-100] * n_prompt + full_ids[n_prompt:] + [-100] * pad_len
        attn_mask = [1] * seq_len + [0] * pad_len
        full_ids = full_ids + [pad_id] * pad_len

        all_input_ids.append(full_ids)
        all_attention_mask.append(attn_mask)
        all_labels.append(labels)

    return DatasetForCausalLM(
        torch.tensor(all_input_ids, dtype=torch.long),
        torch.tensor(all_attention_mask, dtype=torch.long),
        torch.tensor(all_labels, dtype=torch.long),
    )


def _prepare_causal_lm_datasets(
    cfg: DictConfig,
    tokenizer,
) -> tuple[DatasetForCausalLM, DatasetForCausalLM | None]:
    """Causal LM 아키텍처 전용 데이터셋 준비."""
    data_path = _resolve_data_path(cfg.general.data_path)
    use_all_data: bool = getattr(cfg.training, "use_all_data", False)
    data_cfg = getattr(cfg, "data", None)
    use_cleaning: bool = getattr(data_cfg, "use_cleaning", False) if data_cfg else False
    use_length_filter: bool = getattr(data_cfg, "use_length_filter", False) if data_cfg else False
    use_topic: bool = getattr(data_cfg, "use_topic", False) if data_cfg else False
    max_length: int = cfg.tokenizer.encoder_max_len + cfg.tokenizer.decoder_max_len

    train_df = Preprocess.make_set_as_df(os.path.join(data_path, "train.csv"))

    def _apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
        if use_cleaning:
            df = df.copy()
            df["dialogue"] = df["dialogue"].apply(clean_text)
            if "summary" in df.columns:
                df["summary"] = df["summary"].apply(clean_text)
            print("[Preprocess] clean_text 적용 완료")
        if use_length_filter:
            df = filter_by_length(df)
        return df

    train_df = _apply_preprocessing(train_df)

    if use_topic:
        print("[Topic] use_topic=True — 학습 시 topic_mask_prob=0.25 적용")

    if use_all_data:
        val_df = Preprocess.make_set_as_df(os.path.join(data_path, "dev.csv"))
        val_df = _apply_preprocessing(val_df)
        train_df = pd.concat([train_df, val_df], ignore_index=True)
        print(
            "\n" + "=" * 60 + "\n"
            "[최종 제출 모드] use_all_data=True\n"
            f"  train={len(train_df)}건 (train+dev 합산), eval 비활성화\n"
            + "=" * 60 + "\n"
        )
        return _build_causal_lm_dataset(
            train_df, tokenizer, max_length,
            use_topic=use_topic, topic_mask_prob=0.25,
        ), None

    val_df = Preprocess.make_set_as_df(os.path.join(data_path, "dev.csv"))
    if use_cleaning:
        val_df = val_df.copy()
        val_df["dialogue"] = val_df["dialogue"].apply(clean_text)
        if "summary" in val_df.columns:
            val_df["summary"] = val_df["summary"].apply(clean_text)

    # 학습: topic_mask_prob=0.25 / 검증: 0.0 (실제 topic으로 eval)
    train_dataset = _build_causal_lm_dataset(
        train_df, tokenizer, max_length,
        use_topic=use_topic, topic_mask_prob=0.25,
    )
    val_dataset = _build_causal_lm_dataset(
        val_df, tokenizer, max_length,
        use_topic=use_topic, topic_mask_prob=0.0,
    )
    print(f"[Data] train={len(train_dataset)}, val={len(val_dataset)}")
    return train_dataset, val_dataset


def _prepare_datasets(
    cfg: DictConfig,
    preprocessor: Preprocess,
    tokenizer,
) -> tuple[DatasetForSeq2Seq, DatasetForSeq2Seq | None]:
    """데이터셋 준비.

    cfg.training.use_all_data=true 이면 train.csv + dev.csv를 합산해
    전체 데이터로 학습합니다 (최종 제출 전용).
    이 경우 val_dataset=None을 반환해 eval을 완전히 비활성화합니다.
    dev.csv를 eval에 재사용하면 지표가 오염되므로 절대 eval에 포함하지 않습니다.
    """
    if cfg.model.architecture == "causal_lm":
        return _prepare_causal_lm_datasets(cfg, tokenizer)

    data_path = _resolve_data_path(cfg.general.data_path)
    prefix = getattr(cfg.model, "prefix", "")
    use_all_data: bool = getattr(cfg.training, "use_all_data", False)

    # Phase 3 데이터 전처리 config 플래그
    data_cfg = getattr(cfg, "data", None)
    use_cleaning: bool = getattr(data_cfg, "use_cleaning", False) if data_cfg else False
    use_length_filter: bool = getattr(data_cfg, "use_length_filter", False) if data_cfg else False
    use_topic: bool = getattr(data_cfg, "use_topic", False) if data_cfg else False

    def _apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
        """config 플래그에 따라 클리닝·길이 필터를 순서대로 적용합니다."""
        if use_cleaning:
            df = df.copy()
            df["dialogue"] = df["dialogue"].apply(clean_text)
            if "summary" in df.columns:
                df["summary"] = df["summary"].apply(clean_text)
            print(f"[Preprocess] clean_text 적용 완료")
        if use_length_filter:
            df = filter_by_length(df)
        return df

    train_df = preprocessor.make_set_as_df(os.path.join(data_path, "train.csv"))
    train_df = _apply_preprocessing(train_df)

    if use_topic:
        print("[Topic] use_topic=True — 학습 시 topic_mask_prob=0.25 적용")

    enc_max = cfg.tokenizer.encoder_max_len
    dec_max = cfg.tokenizer.decoder_max_len
    tok_kw = dict(
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        return_token_type_ids=False,
    )

    if use_all_data:
        val_df = preprocessor.make_set_as_df(os.path.join(data_path, "dev.csv"))
        val_df = _apply_preprocessing(val_df)
        train_df = pd.concat([train_df, val_df], ignore_index=True)
        print(
            "\n" + "=" * 60 + "\n"
            "[최종 제출 모드] use_all_data=True\n"
            f"  train={len(train_df)}건 (train+dev 합산), eval 비활성화\n"
            "  ※ num_train_epochs를 dev 검증에서 확인한 best epoch으로 설정할 것\n"
            "  ※ Early stopping·eval 지표가 모두 비활성화됩니다\n"
            + "=" * 60 + "\n"
        )
        enc_train, dec_in_train, dec_out_train = preprocessor.make_input(
            train_df, prefix=prefix, use_topic=use_topic, topic_mask_prob=0.25,
        )
        train_dataset = DatasetForSeq2Seq(
            tokenizer(enc_train, truncation=True, max_length=enc_max, **tok_kw),
            tokenizer(dec_in_train, truncation=True, max_length=dec_max, **tok_kw),
            tokenizer(dec_out_train, truncation=True, max_length=dec_max, **tok_kw),
        )
        return train_dataset, None

    val_df = preprocessor.make_set_as_df(os.path.join(data_path, "dev.csv"))
    # val은 길이 필터 제외 (dev 점수 비교 일관성 유지), 클리닝만 적용
    if use_cleaning:
        val_df = val_df.copy()
        val_df["dialogue"] = val_df["dialogue"].apply(clean_text)
        if "summary" in val_df.columns:
            val_df["summary"] = val_df["summary"].apply(clean_text)
    # 학습: topic_mask_prob=0.25 / 검증: 0.0 (실제 topic으로 eval)
    enc_train, dec_in_train, dec_out_train = preprocessor.make_input(
        train_df, prefix=prefix, use_topic=use_topic, topic_mask_prob=0.25,
    )
    enc_val, dec_in_val, dec_out_val = preprocessor.make_input(
        val_df, prefix=prefix, use_topic=use_topic, topic_mask_prob=0.0,
    )

    train_dataset = DatasetForSeq2Seq(
        tokenizer(enc_train, truncation=True, max_length=enc_max, **tok_kw),
        tokenizer(dec_in_train, truncation=True, max_length=dec_max, **tok_kw),
        tokenizer(dec_out_train, truncation=True, max_length=dec_max, **tok_kw),
    )
    val_dataset = DatasetForSeq2Seq(
        tokenizer(enc_val, truncation=True, max_length=enc_max, **tok_kw),
        tokenizer(dec_in_val, truncation=True, max_length=dec_max, **tok_kw),
        tokenizer(dec_out_val, truncation=True, max_length=dec_max, **tok_kw),
    )

    print(f"[Data] train={len(train_dataset)}, val={len(val_dataset)}")
    return train_dataset, val_dataset


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    device: torch.device = get_device()
    cfg_dict: dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore

    # WandB 환경변수는 wandb.init() 이전에 설정해야 적용됩니다.
    os.environ["WANDB_LOG_MODEL"] = "end"
    os.environ["WANDB_WATCH"] = "false"

    # 체크포인트 저장 디렉토리: {checkpoints_root}/yymmdd_run_NNN (run_name에 포함하기 위해 먼저 계산)
    checkpoints_root = _resolve_data_path(cfg.general.checkpoints_root)
    run_id = _next_run_id(checkpoints_root)

    # WandB 초기화 — entity/project는 .env에서 로드, run name에 yymmdd_run_id 포함
    run_name = (
        f"{cfg.model.name}"
        f"_lr{cfg.training.learning_rate}"
        f"_ep{cfg.training.num_train_epochs}"
        f"_{run_id}"
    )
    wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT"),
        name=run_name,
        config=cfg_dict,
        dir=_PROJECT_ROOT,  # Hydra가 CWD를 outputs/로 변경하므로 명시적 지정
    )

    # WandB 메트릭 축 명시 선언 (epoch 기준 x축)
    wandb.define_metric("epoch")
    for _m in ("eval/rouge_1_f1", "eval/rouge_2_f1", "eval/rouge_l_f1", "eval/rouge_combined"):
        wandb.define_metric(_m, step_metric="epoch")

    # 모델 & 토크나이저 로드
    model, tokenizer = load_tokenizer_and_model(cfg, device)

    # 데이터셋 준비
    preprocessor = Preprocess(cfg.tokenizer.bos_token, cfg.tokenizer.eos_token)
    train_dataset, val_dataset = _prepare_datasets(cfg, preprocessor, tokenizer)

    output_dir = os.path.join(checkpoints_root, run_id)
    os.makedirs(output_dir, exist_ok=True)

    # MPS(Apple Silicon)는 fp16 GradScaler를 PyTorch 2.8+ 부터만 지원.
    # MPS 또는 CPU 환경에서는 fp16/bf16을 모두 끄고 fp32로 학습합니다.
    use_fp16 = cfg.training.fp16 and device.type == "cuda"
    use_bf16 = getattr(cfg.training, "bf16", False) and device.type == "cuda"
    if cfg.training.fp16 and device.type != "cuda":
        print(f"[Train] fp16=True 설정이 무시됩니다 (device={device.type}). fp32로 학습합니다.")

    use_all_data: bool = getattr(cfg.training, "use_all_data", False)
    num_train_epochs: int = int(cfg.training.num_train_epochs)
    gen_config = _build_generation_config(cfg, tokenizer=tokenizer, model=model)
    architecture: str = cfg.model.architecture

    if architecture == "causal_lm":
        # Causal LM은 Seq2SeqTrainer 대신 표준 Trainer 사용.
        # predict_with_generate 없이 eval_loss 기반으로 학습합니다.
        common_kw = dict(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            learning_rate=cfg.training.learning_rate,
            per_device_train_batch_size=cfg.training.per_device_train_batch_size,
            warmup_ratio=cfg.training.warmup_ratio,
            weight_decay=cfg.training.weight_decay,
            lr_scheduler_type=cfg.training.lr_scheduler_type,
            optim=cfg.training.optim,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            max_grad_norm=cfg.training.max_grad_norm,
            fp16=use_fp16,
            bf16=use_bf16,
            seed=cfg.training.seed,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_strategy=cfg.training.logging_strategy,
            report_to=cfg.training.report_to,
            label_smoothing_factor=cfg.training.label_smoothing_factor,
        )
        if use_all_data:
            training_args = TrainingArguments(
                **common_kw,
                eval_strategy="no",
                save_strategy="epoch",
                save_total_limit=1,
                load_best_model_at_end=False,
                do_train=True,
                do_eval=False,
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
            )
        else:
            training_args = TrainingArguments(
                **common_kw,
                per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
                eval_strategy=cfg.training.evaluation_strategy,
                save_strategy=cfg.training.save_strategy,
                save_total_limit=cfg.training.save_total_limit,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                load_best_model_at_end=cfg.training.load_best_model_at_end,
                do_train=cfg.training.do_train,
                do_eval=cfg.training.do_eval,
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=cfg.training.early_stopping_patience,
                        early_stopping_threshold=cfg.training.early_stopping_threshold,
                    ),
                    BestCheckpointCallback(output_dir=output_dir, top_k=3, architecture="causal_lm"),
                ],
            )
    elif use_all_data:
        # dev가 학습 데이터에 포함되므로 eval을 완전히 끕니다.
        # BestCheckpointCallback·EarlyStoppingCallback 모두 제거하고
        # Trainer 기본 저장(epoch마다 1개 유지)으로 동작합니다.
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            learning_rate=cfg.training.learning_rate,
            per_device_train_batch_size=cfg.training.per_device_train_batch_size,
            warmup_ratio=cfg.training.warmup_ratio,
            weight_decay=cfg.training.weight_decay,
            lr_scheduler_type=cfg.training.lr_scheduler_type,
            optim=cfg.training.optim,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            max_grad_norm=cfg.training.max_grad_norm,
            eval_strategy="no",
            save_strategy="epoch",
            save_total_limit=1,         # 마지막 epoch 체크포인트 1개만 유지
            fp16=use_fp16,
            bf16=use_bf16,
            load_best_model_at_end=False,
            predict_with_generate=cfg.training.predict_with_generate,
            generation_max_length=cfg.training.generation_max_length,
            generation_config=gen_config,
            do_train=True,
            do_eval=False,
            seed=cfg.training.seed,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_strategy=cfg.training.logging_strategy,
            report_to=cfg.training.report_to,
            label_smoothing_factor=cfg.training.label_smoothing_factor,
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
        )
    else:
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            learning_rate=cfg.training.learning_rate,
            per_device_train_batch_size=cfg.training.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
            warmup_ratio=cfg.training.warmup_ratio,
            weight_decay=cfg.training.weight_decay,
            lr_scheduler_type=cfg.training.lr_scheduler_type,
            optim=cfg.training.optim,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            max_grad_norm=cfg.training.max_grad_norm,
            eval_strategy=cfg.training.evaluation_strategy,
            save_strategy=cfg.training.save_strategy,
            save_total_limit=None,      # BestCheckpointCallback이 top-3 관리
            fp16=use_fp16,
            bf16=use_bf16,
            metric_for_best_model="rouge_combined",
            greater_is_better=True,
            load_best_model_at_end=cfg.training.load_best_model_at_end,
            predict_with_generate=cfg.training.predict_with_generate,
            generation_max_length=cfg.training.generation_max_length,
            generation_config=gen_config,
            do_train=cfg.training.do_train,
            do_eval=cfg.training.do_eval,
            seed=cfg.training.seed,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_strategy=cfg.training.logging_strategy,
            report_to=cfg.training.report_to,
            label_smoothing_factor=cfg.training.label_smoothing_factor,
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=lambda pred: compute_metrics(cfg_dict, tokenizer, pred),
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=cfg.training.early_stopping_patience,
                    early_stopping_threshold=cfg.training.early_stopping_threshold,
                ),
                BestCheckpointCallback(output_dir=output_dir, top_k=3),
            ],
        )

    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()
