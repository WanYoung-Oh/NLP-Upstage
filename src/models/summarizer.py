"""
모델 및 토크나이저 로드 모듈.

architecture 값에 따라 분기:
- bart      → AutoModelForSeq2SeqLM (KoBART 등)
- t5        → AutoModelForSeq2SeqLM + prefix 처리 (KoT5, pko-T5 등)
- causal_lm → AutoModelForCausalLM + PEFT QLoRA (SOLAR 등)
"""

from __future__ import annotations

import json
import os

import torch
from omegaconf import DictConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# T5 등 Unigram: tokenizer.json의 vocab이 dict일 때 KeyError(0) 방지용 몽키패치
_tokenizers_unigram_patch_applied = False


def _patch_unigram_vocab_dict():
    """Unigram vocab이 dict인 경우 리스트로 변환한 뒤 기존 로직을 타도록 convert_to_native_format 보완."""
    from tokenizers import Tokenizer as TokenizerFast

    from transformers import tokenization_utils_tokenizers

    global _tokenizers_unigram_patch_applied
    if _tokenizers_unigram_patch_applied:
        return
    _tokenizers_unigram_patch_applied = True
    TokenizersBackend = tokenization_utils_tokenizers.TokenizersBackend

    _orig_convert = TokenizersBackend.convert_to_native_format

    @classmethod
    def _convert_with_unigram_dict_fix(cls, trust_remote_code=False, **kwargs):
        local_kwargs = dict(kwargs)
        fast_tokenizer_file = local_kwargs.pop("tokenizer_file", None)
        # T5Tokenizer 등: tokenizer_file 있지만 첫 번째 if( tokenizer_object 반환 )는 타지 않고 elif로 진입
        if (
            fast_tokenizer_file is not None
            and os.path.isfile(fast_tokenizer_file)
            and getattr(cls, "model", None) is not None
            and getattr(cls.model, "__name__", None) == "Unigram"
        ):
            with open(fast_tokenizer_file, encoding="utf-8") as f:
                tokenizer_json = json.load(f)
            vocab = tokenizer_json.get("model", {}).get("vocab", None)
            if isinstance(vocab, dict):
                # id 순서 유지한 리스트로 변환 후 아래와 동일한 elif 분기 로직 수행 (vocab[0] 접근 제거)
                vocab = [
                    tuple([k, v])
                    for k, v in sorted(vocab.items(), key=lambda x: x[1])
                ]
                tok_from_file = TokenizerFast.from_file(fast_tokenizer_file)
                local_kwargs["post_processor"] = tok_from_file.post_processor
                local_kwargs["tokenizer_padding"] = tok_from_file.padding
                local_kwargs["tokenizer_truncation"] = tok_from_file.truncation
                if tok_from_file.truncation is not None:
                    local_kwargs["_json_truncation"] = tok_from_file.truncation
                if tok_from_file.padding is not None:
                    local_kwargs["_json_padding"] = tok_from_file.padding
                normalizer_config = tokenizer_json.get("normalizer")
                if normalizer_config:
                    if normalizer_config.get("type") == "Sequence":
                        normalizer_config = normalizer_config["normalizers"]
                    elif not isinstance(normalizer_config, list):
                        normalizer_config = [normalizer_config]
                    for normalizer in normalizer_config:
                        if normalizer.get("type") == "Precompiled" and "precompiled_charsmap" in normalizer:
                            import base64

                            local_kwargs["_spm_precompiled_charsmap"] = base64.b64decode(
                                normalizer["precompiled_charsmap"]
                            )
                            break
                local_kwargs["vocab"] = vocab
                if "merges" in tokenizer_json.get("model", {}):
                    merges = tokenizer_json["model"]["merges"]
                    local_kwargs["merges"] = [
                        tuple(m.split(" ")) if isinstance(m, str) else tuple(m)
                        for m in merges
                    ]
                return local_kwargs
        return _orig_convert(cls, trust_remote_code=trust_remote_code, **kwargs)

    TokenizersBackend.convert_to_native_format = _convert_with_unigram_dict_fix


def load_tokenizer_and_model(
    cfg: DictConfig,
    device: torch.device,
) -> tuple:
    """
    Returns:
        (model, tokenizer)
    """
    architecture: str = cfg.model.architecture
    model_name: str = cfg.model.model_name
    special_tokens: list[str] = list(cfg.tokenizer.special_tokens)

    # T5 계열: tokenizer.json의 Unigram vocab이 dict일 때 KeyError가 나므로 패치 적용 후 일반 로드
    if architecture == "t5":
        _patch_unigram_vocab_dict()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens_dict = {"additional_special_tokens": special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)

    if architecture in ("bart", "t5"):
        revision = getattr(cfg.model, "revision", "main")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=revision)
        model.resize_token_embeddings(len(tokenizer))
        # GPU 메모리 절약 (T5/KoT5 등 대형 seq2seq에서 OOM 방지)
        model.gradient_checkpointing_enable()
        model.to(device)

    elif architecture == "causal_lm":
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import (  # type: ignore
            LoraConfig,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=cfg.model.lora_r,
            lora_alpha=cfg.model.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.resize_token_embeddings(len(tokenizer))
        # causal_lm은 device_map="auto"로 로드되므로 별도 .to(device) 불필요

    else:
        raise ValueError(f"Unsupported architecture: {architecture!r}")

    print(f"[Model] {model_name} ({architecture}) loaded on {device}")
    return model, tokenizer
