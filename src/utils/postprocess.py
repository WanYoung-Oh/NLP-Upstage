import re


def postprocess(text: str, remove_tokens: list | None = None, min_length: int = 10) -> str:
    """생성 텍스트 후처리 파이프라인 (5단계).

    Args:
        text: 생성된 요약문
        remove_tokens: 제거할 특수 토큰 목록
        min_length: 최소 길이 기준 (기본 10자). 이 기준 미달 시 재생성이 필요할 수 있습니다.
                    실제 재생성 여부는 batch_postprocess_with_flags()의 플래그로 확인하세요.
    """
    # 1. 특수 토큰 제거
    if remove_tokens:
        for token in remove_tokens:
            text = text.replace(token, " ")

    # 2. 과도한 공백 정리
    text = re.sub(r"\s+", " ", text).strip()

    # 3. 문장 끝 마침표 보장
    if text and text[-1] not in ".!?":
        text = text + "."

    # 4. 반복 문장 제거 (dict.fromkeys: 삽입 순서 보존 + O(1) 조회)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    text = " ".join(dict.fromkeys(sentences))

    # 5. 최소 길이 보장: 10자 미만이면 원본 반환 (재생성 플래그는 batch 함수에서 제공)
    if len(text) < min_length:
        # 텍스트가 너무 짧으면 그대로 반환하되, 호출자가 플래그를 확인할 수 있도록 함
        return text

    return text


def batch_postprocess(texts: list[str], remove_tokens: list | None = None) -> list[str]:
    return [postprocess(t, remove_tokens) for t in texts]


def batch_postprocess_with_flags(
    texts: list[str],
    remove_tokens: list | None = None,
    min_length: int = 10,
) -> tuple[list[str], list[bool]]:
    """후처리 결과와 함께 재생성 필요 여부 플래그를 반환합니다.

    Args:
        texts: 생성된 요약문 리스트
        remove_tokens: 제거할 특수 토큰 목록
        min_length: 이 길이 미만이면 needs_regen=True 플래그 설정 (기본 10자)

    Returns:
        (processed_texts, needs_regen_flags)
        - processed_texts: 후처리된 텍스트 리스트
        - needs_regen_flags: 각 텍스트의 재생성 필요 여부 (True이면 재생성 권장)

    Example:
        >>> results, flags = batch_postprocess_with_flags(summaries, remove_tokens)
        >>> short_count = sum(flags)
        >>> if short_count > 0:
        ...     print(f"[경고] {short_count}개 요약문이 최소 길이({min_length}자) 미달")
    """
    processed, flags = [], []
    for t in texts:
        result = postprocess(t, remove_tokens, min_length=min_length)
        processed.append(result)
        flags.append(len(result) < min_length)
    return processed, flags
