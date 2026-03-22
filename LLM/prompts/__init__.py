"""
프롬프트 엔지니어링 모듈

한국어 다화자 대화 요약을 위한 프롬프트 템플릿과 유틸리티 함수를 제공합니다.

실측 성능:
- Greedy (1개): ROUGE-1 0.5641
- MBR 8개: ROUGE-1 0.5716 (+0.0075, 약 1.3% 향상)
"""

from .base_prompts import (
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    SYSTEM_PROMPT_TOPIC,
    USER_TEMPLATE_TOPIC,
    formatting_prompts_func,
    create_formatting_func,
    setup_response_only_loss,
)

from .mbr_prompts import (
    PROMPT_VARIANTS,
    get_all_prompt_variants,
    get_prompt_variant,
    create_messages,
)

from .postprocess import (
    postprocess_summary,
    advanced_postprocess,
    batch_postprocess,
)

from .mbr_decoding import (
    mbr_ensemble,
    apply_mbr_to_dataset,
)

from .inference import (
    InferencePipeline,
    quick_inference,
)

from .evaluation import (
    evaluate_rouge,
    evaluate_prompts,
    compare_base_vs_topic,
    evaluate_mbr_ensemble,
)

__version__ = "1.0.0"

__all__ = [
    # Base prompts
    'SYSTEM_PROMPT',
    'USER_TEMPLATE',
    'SYSTEM_PROMPT_TOPIC',
    'USER_TEMPLATE_TOPIC',
    'formatting_prompts_func',
    'create_formatting_func',
    'setup_response_only_loss',
    # MBR prompts
    'PROMPT_VARIANTS',
    'get_all_prompt_variants',
    'get_prompt_variant',
    'create_messages',
    # Postprocess
    'postprocess_summary',
    'advanced_postprocess',
    'batch_postprocess',
    # MBR decoding
    'mbr_ensemble',
    'apply_mbr_to_dataset',
    # Inference
    'InferencePipeline',
    'quick_inference',
    # Evaluation
    'evaluate_rouge',
    'evaluate_prompts',
    'compare_base_vs_topic',
    'evaluate_mbr_ensemble',
]
