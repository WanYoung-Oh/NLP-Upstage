"""
한국어 MeCab 형태소 추출 (mecab-python3: import MeCab).

프로젝트 일부 코드는 예전 `import mecab` / `mecab.MeCab().morphs()` 관례를 쓰지만,
PyPI의 mecab-python3는 `import MeCab` + `MeCab.Tagger()`만 제공하므로 여기서 맞춘다.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List


@lru_cache(maxsize=1)
def _tagger():
    import MeCab
    try:
        return MeCab.Tagger()
    except RuntimeError as e:
        raise ImportError(
            "MeCab 초기화 실패 (사전 경로 없음 등). "
            f"상세: {e}"
        ) from e


class _KoMeCab:
    """`mecab.MeCab()`와 동일하게 `.morphs(text) -> list[str]` 제공."""

    def morphs(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        lines = _tagger().parse(text).rstrip().split("\n")
        out: List[str] = []
        for line in lines:
            if not line or line == "EOS":
                continue
            out.append(line.split("\t", 1)[0])
        return out


def get_mecab() -> _KoMeCab:
    """
    MeCab 분석기 인스턴스. 실패 시 ImportError (호출부에서 원문 폴백).
    """
    try:
        _tagger()
    except ImportError as e:
        raise ImportError("mecab-python3가 필요합니다: pip install mecab-python3") from e
    except OSError as e:
        raise ImportError(
            "MeCab 네이티브 라이브러리를 불러오지 못했습니다. "
            "시스템에 mecab-ko가 설치돼 있는지 확인하세요."
        ) from e
    return _KoMeCab()
