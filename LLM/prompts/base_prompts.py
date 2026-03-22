"""
SFT 학습용 Base 프롬프트 템플릿

한국어 다화자 대화 요약을 위한 기본 프롬프트와 학습 포맷팅 함수를 제공합니다.
검증된 ROUGE-1 0.56+ 성능을 달성한 프롬프트 설계입니다.
"""


# ============================================================================
# Base 프롬프트 (검증된 ROUGE-1 0.56+ 달성)
# ============================================================================

SYSTEM_PROMPT = """당신은 한국어 대화 요약 전문가입니다.
대화에는 #Person1#, #Person2# 등의 화자 태그가 사용됩니다.
요약할 때 이 화자 태그를 그대로 사용하여 누가 무엇을 했는지 명확히 구분해주세요.
핵심 내용만 1~3문장으로 간결하게 요약하세요."""

USER_TEMPLATE = """아래 대화를 읽고 핵심 내용을 요약해주세요.
화자 태그(#Person1# 등)를 유지하세요.

{dialogue}"""


# ============================================================================
# Topic 통합 프롬프트 (개선안)
# ============================================================================

SYSTEM_PROMPT_TOPIC = """당신은 한국어 대화 요약 전문가입니다.
대화의 주제와 화자들의 주요 행동을 파악하여 요약하세요.
화자 태그(#Person1#, #Person2# 등)를 반드시 그대로 사용하세요.
1~3문장으로 간결하게 요약하세요."""

USER_TEMPLATE_TOPIC = """아래 대화의 주제는 "{topic}"입니다.
대화를 읽고 핵심 내용을 요약해주세요.

{dialogue}"""


# ============================================================================
# SFT 학습용 포맷팅 함수
# ============================================================================

def formatting_prompts_func(example, tokenizer, use_topic=False):
    """
    SFT 학습을 위해 데이터를 Chat Template 형식으로 변환
    
    Args:
        example: 데이터셋의 단일 샘플 (dialogue, summary, topic 포함)
        tokenizer: 토크나이저 (apply_chat_template 지원 필요)
        use_topic: Topic 정보 사용 여부
    
    Returns:
        {"text": 포맷팅된 텍스트} 딕셔너리
    """
    # Topic 사용 여부에 따라 프롬프트 선택
    if use_topic and 'topic' in example and example['topic']:
        system = SYSTEM_PROMPT_TOPIC
        user_content = USER_TEMPLATE_TOPIC.format(
            dialogue=example["dialogue"],
            topic=example["topic"]
        )
    else:
        system = SYSTEM_PROMPT
        user_content = USER_TEMPLATE.format(dialogue=example["dialogue"])
    
    # Chat Template 구성
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["summary"]},
    ]
    
    # Tokenizer의 Chat Template 적용
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,  # 학습 시 False 필수
        enable_thinking=False,         # Qwen3 Thinking 모드 비활성화
    )
    
    return {"text": text}


def create_formatting_func(tokenizer, use_topic=False):
    """
    토크나이저를 고정한 포맷팅 함수 생성 (map에 직접 전달 가능)
    
    Args:
        tokenizer: 토크나이저
        use_topic: Topic 정보 사용 여부
    
    Returns:
        포맷팅 함수
    
    Example:
        >>> formatting_func = create_formatting_func(tokenizer, use_topic=False)
        >>> train_dataset = train_dataset.map(formatting_func)
    """
    def _formatting_func(example):
        return formatting_prompts_func(example, tokenizer, use_topic)
    
    return _formatting_func


# ============================================================================
# Response-Only Loss 설정을 위한 헬퍼 함수
# ============================================================================

def get_response_template(model_name="qwen"):
    """
    모델별 Response Template 반환
    
    Args:
        model_name: 모델 종류 ("qwen", "llama", "mistral" 등)
    
    Returns:
        Response template 문자열
    """
    templates = {
        "qwen": "<|im_start|>assistant",
        "llama": "<|start_header_id|>assistant<|end_header_id|>",
        "mistral": "[/INST]",
    }
    
    return templates.get(model_name.lower(), templates["qwen"])


def setup_response_only_loss(tokenizer, model_name="qwen"):
    """
    Response-Only Loss를 위한 DataCollator 설정
    
    Args:
        tokenizer: 토크나이저
        model_name: 모델 종류
    
    Returns:
        DataCollatorForCompletionOnlyLM 인스턴스
    
    Example:
        >>> from trl import DataCollatorForCompletionOnlyLM
        >>> collator = setup_response_only_loss(tokenizer, "qwen")
        >>> trainer = SFTTrainer(
        ...     model=model,
        ...     data_collator=collator,
        ...     ...
        ... )
    """
    from trl import DataCollatorForCompletionOnlyLM
    
    response_template = get_response_template(model_name)
    
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )
    
    return collator


# ============================================================================
# 학습 예시 코드
# ============================================================================

def example_training_setup():
    """
    SFT 학습 설정 예시 (실행 가능한 템플릿)
    
    사용법:
        1. 필요한 라이브러리 import
        2. 모델과 토크나이저 로드
        3. 데이터셋 준비
        4. 아래 코드 참고하여 Trainer 설정
    """
    example_code = '''
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from prompts.base_prompts import (
    create_formatting_func,
    setup_response_only_loss,
)

# 1. 모델 및 토크나이저 로드
model_name = "Qwen/Qwen2.5-14B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
)

# 2. 데이터셋 로드
train_dataset = load_dataset("csv", data_files="train.csv", split="train")

# 3. 포맷팅 함수 적용
formatting_func = create_formatting_func(tokenizer, use_topic=False)
train_dataset = train_dataset.map(formatting_func)

# 4. Response-Only Loss 설정
collator = setup_response_only_loss(tokenizer, model_name="qwen")

# 5. SFT Trainer 구성
training_args = SFTConfig(
    output_dir="./outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    learning_rate=2e-4,
    num_train_epochs=3,
    max_seq_length=2048,
    bf16=True,
    optim="adamw_8bit",
    logging_steps=10,
    save_strategy="epoch",
    packing=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collator,
    tokenizer=tokenizer,
)

# 6. 학습 시작
trainer.train()
    '''
    
    return example_code


if __name__ == "__main__":
    # 프롬프트 확인
    print("=" * 80)
    print("Base 프롬프트")
    print("=" * 80)
    print("\n[System Prompt]")
    print(SYSTEM_PROMPT)
    print("\n[User Template]")
    print(USER_TEMPLATE)
    
    print("\n" + "=" * 80)
    print("Topic 통합 프롬프트")
    print("=" * 80)
    print("\n[System Prompt]")
    print(SYSTEM_PROMPT_TOPIC)
    print("\n[User Template]")
    print(USER_TEMPLATE_TOPIC)
    
    print("\n" + "=" * 80)
    print("학습 예시 코드")
    print("=" * 80)
    print(example_training_setup())
