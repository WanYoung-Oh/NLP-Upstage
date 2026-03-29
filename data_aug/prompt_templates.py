"""
Solar API 호출용 프롬프트 템플릿
"""

SYSTEM_PROMPT = """당신은 한국어 대화 생성 전문가입니다.
주어진 조건에 맞게 자연스러운 한국어 대화를 생성하세요.
반드시 #Person1#:, #Person2#: 형식을 사용하고, 대화만 출력하세요."""


def make_front_prompt(topic: str, summary: str, existing_dialogue: str, n_gen: int) -> str:
    """앞부분 삭제 → 앞부분(도입부) 생성 프롬프트"""
    return f"""주제: {topic}
요약: {summary}

아래는 대화의 뒷부분입니다:
{existing_dialogue}

위 대화 앞에 오는 자연스러운 도입부 대화를 약 {n_gen}턴 생성하세요.
- #Person1#:, #Person2#: 형식을 반드시 사용하세요
- 주제와 요약 내용에 부합해야 합니다
- 뒷부분 대화와 자연스럽게 연결되어야 합니다
- 대화만 출력하고 다른 설명은 쓰지 마세요"""


def make_back_prompt(topic: str, summary: str, existing_dialogue: str, n_gen: int) -> str:
    """뒷부분 삭제 → 뒷부분(마무리) 생성 프롬프트"""
    return f"""주제: {topic}
요약: {summary}

아래는 대화의 앞부분입니다:
{existing_dialogue}

위 대화에 이어지는 자연스러운 뒷부분 대화를 약 {n_gen}턴 생성하세요.
- #Person1#:, #Person2#: 형식을 반드시 사용하세요
- 주제와 요약 내용에 부합해야 합니다
- 기존 대화 흐름과 자연스럽게 연결되어야 합니다
- 대화만 출력하고 다른 설명은 쓰지 마세요"""
