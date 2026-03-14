"""
ROUGE 평가 유틸리티.

평가 모드는 config dict의 metrics.use_korouge 키로 제어합니다:
- false (기본): rouge 라이브러리 — 기존 베이스라인과 동일
- true  (Phase 3+ 권장): korouge-score — 한국어 문자 보존, Java 불필요

config.yaml 예시:
    metrics:
      use_korouge: false

대회 공식 채점 방식 (multi-reference):
  - 예측 1개 vs 정답 3개 → 각 정답에 대해 R1/R2/RL F1 계산
  - 메트릭별 3개 점수 평균 → 세 평균의 합산이 최종 점수
  - `compute_multi_ref_rouge()` / `evaluate_multi_ref()` 사용
"""

from __future__ import annotations


def _decode_and_clean(
    pred,
    tokenizer,
    remove_tokens: list[str],
) -> tuple[list[str], list[str]]:
    """예측값/레이블 디코딩 + 불필요 토큰 제거 + 빈 문자열 방지."""
    predictions = pred.predictions
    labels = pred.label_ids

    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    preds = tokenizer.batch_decode(predictions, clean_up_tokenization_spaces=True)
    golds = tokenizer.batch_decode(labels, clean_up_tokenization_spaces=True)

    for token in remove_tokens:
        preds = [s.replace(token, " ") for s in preds]
        golds = [s.replace(token, " ") for s in golds]

    preds = [s.strip() if s.strip() else "." for s in preds]
    golds = [s.strip() if s.strip() else "." for s in golds]
    return preds, golds


def _rouge_korouge(preds: list[str], golds: list[str]) -> dict[str, float]:
    """korouge-score로 ROUGE 계산 (한국어 문자 보존, Java 불필요)."""
    from korouge_score import rouge_scorer  # type: ignore

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
    r1_f = r2_f = rl_f = 0.0
    for pred, gold in zip(preds, golds):
        scores = scorer.score(gold, pred)
        r1_f += scores["rouge1"].fmeasure
        r2_f += scores["rouge2"].fmeasure
        rl_f += scores["rougeL"].fmeasure
    n = max(len(preds), 1)
    return {"rouge-1": r1_f / n, "rouge-2": r2_f / n, "rouge-l": rl_f / n}


def _rouge_baseline(preds: list[str], golds: list[str]) -> dict[str, float]:
    """기존 rouge 라이브러리로 ROUGE 계산 (베이스라인 호환)."""
    from rouge import Rouge  # type: ignore

    rouge = Rouge()
    results = rouge.get_scores(preds, golds, avg=True)
    return {key: value["f"] for key, value in results.items()}


def _single_rouge(prediction: str, reference: str, use_korouge: bool) -> dict[str, float]:
    """단일 예측/정답 쌍의 ROUGE 계산 내부 헬퍼."""
    pred_safe = prediction.strip() if prediction.strip() else "."
    ref_safe = reference.strip() if reference.strip() else "."
    if use_korouge:
        return _rouge_korouge([pred_safe], [ref_safe])
    return _rouge_baseline([pred_safe], [ref_safe])


def compute_metrics(config: dict, tokenizer, pred) -> dict[str, float]:
    """
    Seq2SeqTrainer의 compute_metrics 콜백 함수.

    config["metrics"]["use_korouge"] = true 이면 korouge-score (한국어 보존),
    false 이면 rouge 라이브러리 (베이스라인 호환) 를 사용합니다.
    """
    remove_tokens: list[str] = config.get("inference", {}).get("remove_tokens", [])
    use_korouge: bool = config.get("metrics", {}).get("use_korouge", False)

    preds, golds = _decode_and_clean(pred, tokenizer, remove_tokens)

    print("-" * 80)
    for i in range(min(3, len(preds))):
        print(f"PRED: {preds[i]}")
        print(f"GOLD: {golds[i]}")
        print("-" * 80)

    mode = "korouge" if use_korouge else "rouge"
    print(f"[Metrics] 평가 모드: {mode}")

    if use_korouge:
        raw = _rouge_korouge(preds, golds)
    else:
        raw = _rouge_baseline(preds, golds)

    r1 = raw.get("rouge-1", 0.0)
    r2 = raw.get("rouge-2", 0.0)
    rl = raw.get("rouge-l", 0.0)
    return {
        "rouge_1_f1": r1,
        "rouge_2_f1": r2,
        "rouge_l_f1": rl,
        "rouge_combined": r1 + r2 + rl,
    }


# ---------------------------------------------------------------------------
# 대회 공식 채점 방식: Multi-reference ROUGE
# ---------------------------------------------------------------------------

def compute_multi_ref_rouge(
    prediction: str,
    references: list[str],
    use_korouge: bool = False,
) -> dict[str, float]:
    """
    단일 예측 vs 다수 정답에 대한 ROUGE 계산 (대회 공식 방식).

    각 정답에 대해 R1/R2/RL F1을 계산하고 메트릭별로 평균한 뒤 합산합니다.

    Args:
        prediction: 모델 예측 요약문
        references: 정답 요약문 리스트 (대회 기준 3개)
        use_korouge: True이면 korouge-score 사용 (한국어 보존)

    Returns:
        {
            "rouge_1_f1": mean(R1 across refs),
            "rouge_2_f1": mean(R2 across refs),
            "rouge_l_f1": mean(RL across refs),
            "rouge_combined": 세 메트릭 평균의 합산 (대회 최종 점수 단위)
        }

    Example:
        >>> score = compute_multi_ref_rouge("김씨가 약속을 잡았다.", ["김씨가 약속을 잡았다.", ...])
        >>> # 대회 점수: score["rouge_combined"] ≈ 47.12 스케일
    """
    if not references:
        return {"rouge_1_f1": 0.0, "rouge_2_f1": 0.0, "rouge_l_f1": 0.0, "rouge_combined": 0.0}

    r1_scores, r2_scores, rl_scores = [], [], []
    for ref in references:
        if not ref.strip():
            continue
        scores = _single_rouge(prediction, ref, use_korouge)
        r1_scores.append(scores.get("rouge-1", 0.0))
        r2_scores.append(scores.get("rouge-2", 0.0))
        rl_scores.append(scores.get("rouge-l", 0.0))

    if not r1_scores:
        return {"rouge_1_f1": 0.0, "rouge_2_f1": 0.0, "rouge_l_f1": 0.0, "rouge_combined": 0.0}

    r1 = sum(r1_scores) / len(r1_scores)
    r2 = sum(r2_scores) / len(r2_scores)
    rl = sum(rl_scores) / len(rl_scores)
    return {
        "rouge_1_f1": r1,
        "rouge_2_f1": r2,
        "rouge_l_f1": rl,
        "rouge_combined": r1 + r2 + rl,
    }


def evaluate_multi_ref(
    predictions: list[str],
    multi_refs: list[list[str]],
    use_korouge: bool = False,
) -> dict[str, float]:
    """
    배치 단위 다중 정답 ROUGE 평가 (대회 공식 채점 방식).

    Args:
        predictions: 모델 예측 요약문 리스트 (N개)
        multi_refs: 샘플별 정답 리스트의 리스트 (N × n_refs)
                    예: [["ref1_A", "ref1_B", "ref1_C"], ["ref2_A", ...], ...]
        use_korouge: True이면 korouge-score 사용

    Returns:
        전체 샘플의 메트릭 평균:
        {
            "rouge_1_f1": float,
            "rouge_2_f1": float,
            "rouge_l_f1": float,
            "rouge_combined": float   ← 대회 최종 점수 (≈ 47~60 스케일)
        }

    Notes:
        - dev.csv는 정답 1개 → single-reference 평가와 동일
        - 대회 평가 데이터는 정답 3개 → multi_refs에 3개씩 넣어 사용
        - 로컬 dev 점수는 대회 점수보다 낮게 나올 수 있음
    """
    if not predictions:
        return {"rouge_1_f1": 0.0, "rouge_2_f1": 0.0, "rouge_l_f1": 0.0, "rouge_combined": 0.0}

    r1_all, r2_all, rl_all = [], [], []
    for pred, refs in zip(predictions, multi_refs):
        per_sample = compute_multi_ref_rouge(pred, refs, use_korouge=use_korouge)
        r1_all.append(per_sample["rouge_1_f1"])
        r2_all.append(per_sample["rouge_2_f1"])
        rl_all.append(per_sample["rouge_l_f1"])

    n = max(len(r1_all), 1)
    r1 = sum(r1_all) / n
    r2 = sum(r2_all) / n
    rl = sum(rl_all) / n
    return {
        "rouge_1_f1": r1,
        "rouge_2_f1": r2,
        "rouge_l_f1": rl,
        "rouge_combined": r1 + r2 + rl,
    }


def compare_rouge_modes(
    predictions: list[str],
    references: list[str],
) -> dict[str, dict[str, float]]:
    """
    두 평가 모드의 점수를 동시에 계산해 비교합니다 (Phase 3 분석용).

    Returns:
        {"baseline": {...}, "korouge": {...}}
    """
    baseline = _rouge_baseline(
        [s if s.strip() else "." for s in predictions],
        [s if s.strip() else "." for s in references],
    )
    korouge = _rouge_korouge(
        [s if s.strip() else "." for s in predictions],
        [s if s.strip() else "." for s in references],
    )
    return {"baseline": baseline, "korouge": korouge}
