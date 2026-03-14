"""
모델 및 토크나이저 로드 모듈.

architecture 값에 따라 분기:
- bart      → AutoModelForSeq2SeqLM (KoBART 등)
- t5        → AutoModelForSeq2SeqLM + prefix 처리 (KoT5, pko-T5 등)
- causal_lm → AutoModelForCausalLM + PEFT QLoRA (SOLAR 등)
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens_dict = {"additional_special_tokens": special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)

    if architecture in ("bart", "t5"):
        revision = getattr(cfg.model, "revision", "main")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=revision)
        model.resize_token_embeddings(len(tokenizer))
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
