"""
MBR 앙상블용 프롬프트 변형 모듈

MBR(Minimum Bayes Risk) 디코딩을 위한 프롬프트 변형을 제공합니다.
활성 7종: base, topic, narrative, qa_style, gold_mimic, observer, length_constrained
비활성 4종(주석): abstract, oneshot, threeshot, base_copy
다양한 스타일의 프롬프트를 통해 앙상블 효과를 극대화합니다.

실측 성능:
- Greedy (1개): ROUGE-1 0.5641
- MBR 8개: ROUGE-1 0.5716 (+0.0075, 약 1.3% 향상)
"""


# ============================================================================
# 변형 1: Base (dev_save) - 기준선
# ============================================================================

PROMPT_1_BASE = {
    "name": "base",
    "system": """당신은 한국어 대화 요약 전문가입니다.
대화에는 #Person1#, #Person2# 등의 화자 태그가 사용됩니다.
요약할 때 이 화자 태그를 그대로 사용하여 누가 무엇을 했는지 명확히 구분해주세요.
핵심 내용만 1~3문장으로 간결하게 요약하세요.
대화에 등장하는 사람 이름, 장소, 제품명 등 고유명사는 원문 그대로(영어면 영어, 한글이면 한글) 사용하세요.""",
    "user": "아래 대화를 읽고 핵심 내용을 요약해주세요.\n화자 태그(#Person1# 등)를 유지하세요.\n\n{dialogue}",
    "description": "가장 안정적인 기본 프롬프트"
}


# ============================================================================
# 변형 2: 추상적 스타일 (abstract)
# ============================================================================

PROMPT_2_ABSTRACT = {
    "name": "abstract",
    "system": """당신은 한국어 대화 요약 전문가입니다.
대화의 주요 주제와 화자들의 행동을 요약하세요.
#Person1#, #Person2# 등 화자 태그를 반드시 사용하고,
'~에 대해 이야기한다', '~을 요청한다' 같은 표현을 활용하세요.
1~2문장으로 간결하게 요약하세요.""",
    "user": "다음 대화를 요약하세요.\n\n{dialogue}",
    "description": "추상적 동사 사용 권장 (논의하다, 제안하다 등)"
}


# ============================================================================
# 변형 3: 1-shot Few-shot (goldstyle)
# ============================================================================

PROMPT_3_ONESHOT = {
    "name": "oneshot",
    "system": """당신은 한국어 대화 요약 전문가입니다.
주어진 대화를 읽고 핵심 내용을 1~2문장으로 요약하세요.

규칙:
- 화자 태그(#Person1#, #Person2# 등)를 반드시 그대로 사용하세요.
- 불필요한 세부사항은 생략하고 핵심 행동/결정/결과만 포함하세요.
- 요약은 반드시 완전한 문장으로 끝나야 합니다.

[예시]
대화: #Person1#: 이것은 좋은 기본 컴퓨터 패키지입니다. #Person2#: 모뎀도 포함되어 있나요? #Person1#: 네, 내장 모뎀이 있습니다. #Person2#: 좋습니다. 구매하겠습니다.
요약: #Person1#은 기본 컴퓨터 패키지를 #Person2#에게 보여주고, #Person2#는 구매하기로 한다.""",
    "user": "다음 대화를 요약하세요.\n\n{dialogue}",
    "description": "1-shot 예시로 출력 형식 일관성 향상"
}


# ============================================================================
# 변형 4: Topic 활용 (topic_integrated)
# ============================================================================

PROMPT_4_TOPIC = {
    "name": "topic",
    "system": """당신은 한국어 대화 요약 전문가입니다.
대화의 주제와 화자들의 주요 행동을 파악하여 요약하세요.
화자 태그(#Person1#, #Person2# 등)를 반드시 사용하세요.
1~3문장으로 간결하게 요약하세요.
대화에 등장하는 사람 이름, 장소, 제품명 등 고유명사는 원문 그대로(영어면 영어, 한글이면 한글) 사용하세요.""",
    "user": "아래 대화의 주제는 \"{topic}\"입니다.\n대화를 읽고 핵심 내용을 요약해주세요.\n\n{dialogue}",
    "description": "Topic 힌트 제공으로 맥락 파악 용이"
}


# ============================================================================
# 변형 5: 서술형 스타일 (narrative)
# ============================================================================

PROMPT_5_NARRATIVE = {
    "name": "narrative",
    "system": """당신은 한국어 대화 요약 전문가입니다.
대화를 읽고 누가 무엇을 했는지 명확하게 서술하세요.
화자 태그(#Person1#, #Person2#)를 반드시 유지하세요.
1~3문장으로 핵심 내용만 요약하세요.
대화에 등장하는 사람 이름, 장소, 제품명 등 고유명사는 원문 그대로(영어면 영어, 한글이면 한글) 사용하세요.""",
    "user": "다음 대화의 내용을 간단히 설명해주세요.\n\n{dialogue}",
    "description": "서술형 스타일로 행동 중심 요약"
}


# ============================================================================
# 변형 6: 질의응답 스타일 (qa_style)
# ============================================================================

PROMPT_6_QA = {
    "name": "qa_style",
    "system": """당신은 한국어 대화 요약 전문가입니다.
대화를 분석하여 화자들이 무엇을 논의하고 어떤 결정을 내렸는지 요약하세요.
#Person1#, #Person2# 등의 화자 표기를 그대로 유지하세요.
간결하게 1~2문장으로 작성하세요.
대화에 등장하는 사람 이름, 장소, 제품명 등 고유명사는 원문 그대로(영어면 영어, 한글이면 한글) 사용하세요.""",
    "user": "이 대화에서 무슨 일이 일어났나요?\n\n{dialogue}",
    "description": "질의응답 형식으로 대화 내용 파악"
}


# ============================================================================
# 변형 7: 3-shot Few-shot (goldstyle_v3)
# ============================================================================

PROMPT_7_THREESHOT = {
    "name": "threeshot",
    "system": """당신은 한국어 대화 요약 전문가입니다.
아래 예시를 참고하여 대화를 요약하세요.

[예시 1]
대화: #Person1#: 이 버스가 센트럴 파크로 가나요? #Person2#: 네, 맞습니다. #Person1#: 언제쯤 도착하나요? #Person2#: 두 정거장만 더 가시면 돼요.
요약: #Person1#이 센트럴 파크로 가는 버스를 타고 있으며, #Person2#가 내릴 정류장을 안내합니다.

[예시 2]
대화: #Person1#: 새 드레스 어때? #Person2#: 정말 예쁘다. 면접 가는 거야? #Person1#: 아니, 학교에서 강연해.
요약: #Person1#은 새 드레스를 샀으며, 학교 강연을 위한 것이다.

[예시 3]
대화: #Person1#: 주문하시겠어요? #Person2#: 네, 정식으로 할게요. #Person1#: 오늘 양고기 챱도 추천합니다. #Person2#: 좋아요, 그걸로 할게요.
요약: #Person2#는 #Person1#의 추천으로 양고기 챱을 주문합니다.""",
    "user": "다음 대화를 요약하세요.\n\n{dialogue}",
    "description": "3-shot 예시로 강력한 형식 학습"
}


# ============================================================================
# 변형 8: Base 복제 (안정성 확보)
# ============================================================================

PROMPT_8_BASE_COPY = {
    "name": "base_copy",
    "system": PROMPT_1_BASE["system"],
    "user": PROMPT_1_BASE["user"],
    "description": "Base와 동일 (과도한 다양성 방지)"
}


# ============================================================================
# 변형 A: Gold 패턴 모방형 - 실제 데이터 패턴을 명시적으로 학습
# ============================================================================

PROMPT_GOLD_MIMIC = {
    "name": "gold_mimic",
    "system": """당신은 한국어 대화 요약 전문가입니다.
다음 규칙을 엄격히 따르세요:
- #Person1#, #Person2# 태그로 시작하세요.
- '~에 대해 이야기한다', '~을 설명한다', '~을 제안한다', '~을 요청한다' 같은 표현을 사용하세요.
- 반드시 1~2문장, 마침표로 끝내세요.
- 대화에 등장하는 사람 이름, 장소, 제품명 등 고유명사는 원문 그대로(영어면 영어, 한글이면 한글) 사용하세요.""",
    "user": "아래 대화를 1~2문장으로 요약하세요.\n\n{dialogue}",
    "description": "Gold 정답 패턴을 명시적으로 규칙화한 프롬프트"
}


# ============================================================================
# 변형 B: 역할 강화형 - 제3자 관찰자 시점 강조
# ============================================================================

PROMPT_OBSERVER = {
    "name": "observer",
    "system": """당신은 대화 내용을 기록하는 제3자 관찰자입니다.
화자 태그(#Person1#, #Person2#)를 사용하여 누가 무엇을 했는지 객관적으로 기술하세요.
핵심 행동과 결과만 1~2문장으로 작성하세요.
대화에 등장하는 사람 이름, 장소, 제품명 등 고유명사는 원문 그대로(영어면 영어, 한글이면 한글) 사용하세요.""",
    "user": "다음 대화에서 일어난 일을 요약하세요.\n\n{dialogue}",
    "description": "제3자 관찰자 시점으로 객관적 서술 유도"
}


# ============================================================================
# 변형 C: 길이 제약 명시형
# ============================================================================

PROMPT_LENGTH_CONSTRAINED = {
    "name": "length_constrained",
    "system": """당신은 한국어 대화 요약 전문가입니다.
화자 태그(#Person1#, #Person2#)를 반드시 사용하세요.
요약은 50~150자 이내로, 반드시 완전한 문장으로 작성하세요.
대화에 등장하는 사람 이름, 장소, 제품명 등 고유명사는 원문 그대로(영어면 영어, 한글이면 한글) 사용하세요.""",
    "user": "아래 대화의 핵심을 요약하세요.\n\n{dialogue}",
    "description": "명시적 길이 제약으로 간결한 요약 유도"
}


# ============================================================================
# 프롬프트 변형 딕셔너리
# ============================================================================


PROMPT_VARIANTS = {
    "base": PROMPT_1_BASE,
    # "abstract": PROMPT_2_ABSTRACT,         # 4-B 실험: 단독 0.7176 → 하위권, MBR 노이즈
    # "oneshot": PROMPT_3_ONESHOT,           # 4-B 실험: 단독 0.7130 → 하위권, MBR 노이즈
    "topic": PROMPT_4_TOPIC,
    # "narrative": PROMPT_5_NARRATIVE,
    "qa_style": PROMPT_6_QA,
    # "threeshot": PROMPT_7_THREESHOT,       # 4-B 실험: 단독 0.6993 → 최하위, MBR 노이즈
    # "base_copy": PROMPT_8_BASE_COPY,  # base와 동일 → 다양성 기여 없음, 신규 3종 추가로 불필요
    "gold_mimic": PROMPT_GOLD_MIMIC,
    # "observer": PROMPT_OBSERVER,
    # "length_constrained": PROMPT_LENGTH_CONSTRAINED,
}


# ============================================================================
# 유틸리티 함수
# ============================================================================

def get_all_prompt_variants():
    """
    모든 프롬프트 변형 반환
    
    Returns:
        {name: {"system": str, "user": str, "description": str}} 딕셔너리
    """
    return PROMPT_VARIANTS


def get_prompt_variant(name):
    """
    특정 프롬프트 변형 반환
    
    Args:
        name: 프롬프트 이름 ("base", "abstract", "oneshot" 등)
    
    Returns:
        {"system": str, "user": str, "description": str} 딕셔너리
        존재하지 않으면 None
    """
    return PROMPT_VARIANTS.get(name)


def format_prompt(variant, dialogue, topic=""):
    """
    프롬프트 변형을 실제 대화에 적용
    
    Args:
        variant: 프롬프트 변형 딕셔너리 또는 이름
        dialogue: 대화 텍스트
        topic: 주제 (topic 프롬프트 사용 시 필요)
    
    Returns:
        {"system": str, "user": str} 딕셔너리
    
    Example:
        >>> variant = get_prompt_variant("base")
        >>> messages = format_prompt(variant, "#Person1#: 안녕...")
        >>> print(messages["system"])
        >>> print(messages["user"])
    """
    # 문자열이면 프롬프트 변형 가져오기
    if isinstance(variant, str):
        variant = get_prompt_variant(variant)
        if variant is None:
            raise ValueError(f"Unknown prompt variant: {variant}")
    
    # User 프롬프트에 dialogue와 topic 적용
    user_content = variant["user"].format(dialogue=dialogue, topic=topic)
    
    return {
        "system": variant["system"],
        "user": user_content,
    }


def create_messages(variant, dialogue, topic=""):
    """
    Chat Template용 messages 리스트 생성
    
    Args:
        variant: 프롬프트 변형 딕셔너리 또는 이름
        dialogue: 대화 텍스트
        topic: 주제
    
    Returns:
        [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
    
    Example:
        >>> messages = create_messages("base", "#Person1#: 안녕...")
        >>> text = tokenizer.apply_chat_template(messages, ...)
    """
    prompt = format_prompt(variant, dialogue, topic)
    
    return [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]},
    ]


def get_prompt_statistics():
    """
    프롬프트 변형 통계 반환
    
    Returns:
        통계 정보 딕셔너리
    """
    stats = {
        "total_variants": len(PROMPT_VARIANTS),
        "variants": {},
    }
    
    for name, variant in PROMPT_VARIANTS.items():
        stats["variants"][name] = {
            "system_length": len(variant["system"]),
            "user_template_length": len(variant["user"]),
            "description": variant["description"],
        }
    
    return stats


# ============================================================================
# 예상 선택 빈도 (실측 기반)
# ============================================================================

EXPECTED_SELECTION_FREQUENCY = {
    "base": "30-35%",               # 가장 안정적
    "base_copy": "30-35%",          # base와 합산
    "abstract": "18-20%",           # 스타일 다양성
    "oneshot": "15-17%",            # 형식 일관성
    "topic": "12-15%",              # 맥락 활용
    "narrative": "8-10%",           # 서술 스타일
    "qa_style": "5-8%",             # 질의 스타일 (단독 최고: 0.7514)
    "threeshot": "3-5%",            # 과도한 예시
    "gold_mimic": "미측정",          # Gold 패턴 모방형 (신규)
    "observer": "미측정",            # 제3자 관찰자형 (신규)
    "length_constrained": "미측정",  # 길이 제약형 (신규)
}


if __name__ == "__main__":
    # 프롬프트 변형 확인
    print("=" * 80)
    print("MBR 앙상블용 프롬프트 변형 (활성 7개 / 전체 11개)")
    print("=" * 80)
    
    for i, (name, variant) in enumerate(PROMPT_VARIANTS.items(), 1):
        print(f"\n[변형 {i}: {name}]")
        print(f"설명: {variant['description']}")
        print(f"예상 선택 빈도: {EXPECTED_SELECTION_FREQUENCY.get(name, 'N/A')}")
        print(f"\n시스템 프롬프트 길이: {len(variant['system'])} 문자")
        print(f"사용자 템플릿 길이: {len(variant['user'])} 문자")
        
        # 샘플 대화로 테스트
        sample_dialogue = "#Person1#: 안녕하세요. #Person2#: 반갑습니다."
        messages = create_messages(name, sample_dialogue, topic="인사")
        print(f"\n[생성된 Messages 미리보기]")
        print(f"System: {messages[0]['content'][:100]}...")
        print(f"User: {messages[1]['content'][:100]}...")
    
    print("\n" + "=" * 80)
    print("통계")
    print("=" * 80)
    stats = get_prompt_statistics()
    print(f"총 변형 수: {stats['total_variants']}")
    print(f"\n변형별 상세:")
    for name, info in stats["variants"].items():
        print(f"  {name:15s}: System {info['system_length']:4d}자, "
              f"User {info['user_template_length']:3d}자 - {info['description']}")
