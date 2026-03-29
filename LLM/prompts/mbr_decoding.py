"""
MBR (Minimum Bayes Risk) 디코딩 알고리즘

MBR 디코딩을 통해 여러 후보 요약 중 최적의 요약을 선택합니다.
ROUGE 기반 pairwise 유사도를 계산하여 가장 높은 기대값을 가진 요약을 반환합니다.

실측 성능:
- Greedy (1개): ROUGE-1 0.5641
- MBR 8개: ROUGE-1 0.5716 (+0.0075, 약 1.3% 향상)

원리:
    여러 후보 요약 중 ROUGE 기대값이 최대인 요약 선택
    y* = argmax_y (1/|H| * sum_{y' in H} ROUGE(y, y'))
"""

from typing import List, Tuple, Dict, Optional
from tqdm import tqdm


def mbr_ensemble(candidates, use_mecab=True, metric="rouge-1"):
    """
    MBR 디코딩으로 최종 요약 선택
    
    Args:
        candidates: [(prompt_name, summary_text), ...] 형태의 리스트
        use_mecab: Mecab 형태소 분석 사용 여부 (대회 평가 기준)
        metric: 사용할 ROUGE 메트릭 ("rouge-1", "rouge-2", "rouge-l")
    
    Returns:
        최종 선택된 요약 텍스트
    
    원리:
        각 후보에 대해 다른 모든 후보와의 ROUGE 점수를 계산하고,
        평균 ROUGE 점수가 가장 높은 후보를 선택합니다.
        이는 "다수결" 방식으로, 여러 프롬프트가 비슷하게 요약한 내용이
        가장 신뢰할 수 있다는 가정에 기반합니다.
    
    Example:
        >>> candidates = [
        ...     ("base", "#Person1#은 #Person2#에게 인사한다."),
        ...     ("abstract", "#Person1#은 #Person2#와 인사를 나눈다."),
        ...     ("oneshot", "#Person1#이 #Person2#에게 인사했다."),
        ... ]
        >>> best = mbr_ensemble(candidates, use_mecab=True)
        >>> print(best)
    """
    try:
        from rouge import Rouge
    except ImportError:
        raise ImportError("rouge 패키지가 필요합니다: pip install rouge")
    
    rouge = Rouge()
    
    # 1. Mecab 형태소 분석 (대회 평가 기준)
    if use_mecab:
        try:
            from .mecab_ko import get_mecab
            m = get_mecab()
            cand_morphs = []
            for name, text in candidates:
                if text and text.strip():
                    morphs = " ".join(m.morphs(text))
                    cand_morphs.append(morphs if morphs else "빈요약")
                else:
                    cand_morphs.append("빈요약")
        except ImportError:
            print("Warning: MeCab을 사용할 수 없습니다. 원본 텍스트를 사용합니다.")
            cand_morphs = [c[1] if c[1].strip() else "빈요약" for c in candidates]
    else:
        cand_morphs = [c[1] if c[1].strip() else "빈요약" for c in candidates]
    
    # 2. Pairwise ROUGE 계산
    best_score = -1
    best_idx = 0
    n = len(candidates)
    
    for j in range(n):
        avg_rouge = 0
        count = 0
        
        for k in range(n):
            if j != k:
                try:
                    # ROUGE 점수 계산
                    scores = rouge.get_scores([cand_morphs[j]], [cand_morphs[k]])[0]
                    avg_rouge += scores[metric]["f"]
                    count += 1
                except Exception as e:
                    # 빈 요약 등 예외 처리
                    pass
        
        if count > 0:
            avg_rouge /= count
        
        # 최고 점수 업데이트
        if avg_rouge > best_score:
            best_score = avg_rouge
            best_idx = j
    
    return candidates[best_idx][1]  # 원본 텍스트 반환 (형태소 아님)


def apply_mbr_to_dataset(test_df, all_predictions, use_mecab=True, metric="rouge-1", verbose=True):
    """
    데이터셋 전체에 MBR 앙상블 적용
    
    Args:
        test_df: 테스트 데이터프레임 (또는 길이)
        all_predictions: {prompt_name: [pred1, pred2, ...]} 딕셔너리
        use_mecab: Mecab 형태소 분석 사용 여부
        metric: 사용할 ROUGE 메트릭
        verbose: 진행 상황 및 통계 출력 여부
    
    Returns:
        MBR로 선택된 최종 요약 리스트
    
    Example:
        >>> all_preds = {
        ...     "base": ["요약1-base", "요약2-base", ...],
        ...     "abstract": ["요약1-abstract", "요약2-abstract", ...],
        ...     ...
        ... }
        >>> final = apply_mbr_to_dataset(test_df, all_preds, use_mecab=True)
    """
    model_names = list(all_predictions.keys())
    n_samples = len(test_df) if hasattr(test_df, '__len__') else test_df
    
    mbr_preds = []
    model_selected = {name: 0 for name in model_names}
    
    # 진행 바 설정
    iterator = range(n_samples)
    if verbose:
        iterator = tqdm(iterator, desc="MBR Ensemble")
    
    for i in iterator:
        # 이 샘플에 대한 모든 후보 수집
        candidates = [(name, all_predictions[name][i]) for name in model_names]
        
        # MBR로 최적 요약 선택
        selected = mbr_ensemble(candidates, use_mecab=use_mecab, metric=metric)
        mbr_preds.append(selected)
        
        # 어떤 프롬프트가 선택되었는지 통계
        for name, text in candidates:
            if text == selected:
                model_selected[name] += 1
                break
    
    # 선택 빈도 출력
    if verbose:
        print("\n" + "=" * 80)
        print("MBR 앙상블 결과 - 모델 선택 빈도")
        print("=" * 80)
        for name, count in sorted(model_selected.items(), key=lambda x: -x[1]):
            percentage = 100 * count / n_samples
            bar = "█" * int(percentage / 2)
            print(f"  {name:20s}: {count:4d} ({percentage:5.1f}%) {bar}")
        print("=" * 80)
    
    return mbr_preds


def apply_mbr_to_dataset_multi(test_df, all_predictions, use_mecab=True, verbose=True, weights=None):
    """ROUGE-1/2/L 평균(mbr_multi_metric)으로 샘플별 최적 후보 선택."""
    model_names = list(all_predictions.keys())
    n_samples = len(test_df) if hasattr(test_df, "__len__") else test_df

    mbr_preds = []
    model_selected = {name: 0 for name in model_names}

    iterator = range(n_samples)
    if verbose:
        iterator = tqdm(iterator, desc="MBR Ensemble (ROUGE multi)")

    for i in iterator:
        candidates = [(name, all_predictions[name][i]) for name in model_names]
        selected = mbr_multi_metric(candidates, use_mecab=use_mecab, weights=weights)
        mbr_preds.append(selected)
        for name, text in candidates:
            if text == selected:
                model_selected[name] += 1
                break

    if verbose:
        print("\n" + "=" * 80)
        print("MBR 앙상블 결과 - 모델 선택 빈도 (ROUGE multi)")
        print("=" * 80)
        for name, count in sorted(model_selected.items(), key=lambda x: -x[1]):
            percentage = 100 * count / n_samples
            bar = "█" * int(percentage / 2)
            print(f"  {name:40s}: {count:4d} ({percentage:5.1f}%) {bar}")
        print("=" * 80)

    return mbr_preds


def mbr_with_weights(candidates, weights=None, use_mecab=True, metric="rouge-1"):
    """
    가중치를 적용한 MBR 디코딩
    
    특정 프롬프트에 더 높은 가중치를 부여하여 선택 확률을 조정할 수 있습니다.
    
    Args:
        candidates: [(prompt_name, summary_text), ...] 형태의 리스트
        weights: {prompt_name: weight} 딕셔너리 (None이면 동일 가중치)
        use_mecab: Mecab 형태소 분석 사용 여부
        metric: 사용할 ROUGE 메트릭
    
    Returns:
        최종 선택된 요약 텍스트
    
    Example:
        >>> weights = {"base": 1.5, "abstract": 1.2, "oneshot": 1.0}
        >>> best = mbr_with_weights(candidates, weights=weights)
    """
    try:
        from rouge import Rouge
    except ImportError:
        raise ImportError("rouge 패키지가 필요합니다: pip install rouge")
    
    rouge = Rouge()
    
    # 기본 가중치 설정
    if weights is None:
        weights = {name: 1.0 for name, _ in candidates}
    
    # Mecab 형태소 분석
    if use_mecab:
        try:
            from .mecab_ko import get_mecab
            m = get_mecab()
            cand_morphs = []
            for name, text in candidates:
                if text and text.strip():
                    morphs = " ".join(m.morphs(text))
                    cand_morphs.append(morphs if morphs else "빈요약")
                else:
                    cand_morphs.append("빈요약")
        except ImportError:
            cand_morphs = [c[1] if c[1].strip() else "빈요약" for c in candidates]
    else:
        cand_morphs = [c[1] if c[1].strip() else "빈요약" for c in candidates]
    
    # Pairwise ROUGE 계산 (가중치 적용)
    best_score = -1
    best_idx = 0
    n = len(candidates)
    
    for j in range(n):
        weighted_rouge = 0
        total_weight = 0
        
        for k in range(n):
            if j != k:
                try:
                    scores = rouge.get_scores([cand_morphs[j]], [cand_morphs[k]])[0]
                    weight = weights.get(candidates[k][0], 1.0)
                    weighted_rouge += scores[metric]["f"] * weight
                    total_weight += weight
                except:
                    pass
        
        if total_weight > 0:
            weighted_rouge /= total_weight

        if weighted_rouge > best_score:
            best_score = weighted_rouge
            best_idx = j
    
    return candidates[best_idx][1]


def mbr_multi_metric(candidates, use_mecab=True, metrics=None, weights=None):
    """
    여러 ROUGE 메트릭을 종합한 MBR 디코딩
    
    ROUGE-1, ROUGE-2, ROUGE-L을 모두 고려하여 최적 후보를 선택합니다.
    
    Args:
        candidates: [(prompt_name, summary_text), ...] 형태의 리스트
        use_mecab: Mecab 형태소 분석 사용 여부
        metrics: 사용할 메트릭 리스트 (None이면 ["rouge-1", "rouge-2", "rouge-l"])
    
    Returns:
        최종 선택된 요약 텍스트
    """
    if metrics is None:
        metrics = ["rouge-1", "rouge-2", "rouge-l"]
    
    try:
        from rouge import Rouge
    except ImportError:
        raise ImportError("rouge 패키지가 필요합니다: pip install rouge")
    
    rouge = Rouge()
    
    # Mecab 형태소 분석
    if use_mecab:
        try:
            from .mecab_ko import get_mecab
            m = get_mecab()
            cand_morphs = []
            for name, text in candidates:
                if text and text.strip():
                    morphs = " ".join(m.morphs(text))
                    cand_morphs.append(morphs if morphs else "빈요약")
                else:
                    cand_morphs.append("빈요약")
        except ImportError:
            cand_morphs = [c[1] if c[1].strip() else "빈요약" for c in candidates]
    else:
        cand_morphs = [c[1] if c[1].strip() else "빈요약" for c in candidates]
    
    # Pairwise ROUGE 계산 (모든 메트릭 평균)
    best_score = -1
    best_idx = 0
    n = len(candidates)
    
    for j in range(n):
        total_rouge = 0.0
        total_weight = 0.0

        for k in range(n):
            if j != k:
                try:
                    scores = rouge.get_scores([cand_morphs[j]], [cand_morphs[k]])[0]
                    # 모든 메트릭의 평균
                    metric_avg = sum(scores[m]["f"] for m in metrics) / len(metrics)
                    ref_w = float(weights.get(candidates[k][0], 1.0)) if weights else 1.0
                    total_rouge += metric_avg * ref_w
                    total_weight += ref_w
                except:
                    pass

        if total_weight > 0:
            avg_rouge = total_rouge / total_weight
        else:
            avg_rouge = 0
        
        if avg_rouge > best_score:
            best_score = avg_rouge
            best_idx = j
    
    return candidates[best_idx][1]


def mbr_ensemble_asymmetric(
    candidates,
    reference_keys,
    use_mecab=True,
    metric="rouge-1",
    multi_metrics=None,
    weights=None,
):
    """
    비대칭 MBR: hypothesis 풀 H 전체에서 고르고, utility는 reference 풀 R에 대해서만 평균.

    y* = argmax_{y in H} (1/|R|) * sum_{r in R} w_r * U(y, r)

    Args:
        candidates: [(key, text), ...] — 전체 후보 (H)
        reference_keys: R에 포함할 candidate key 집합. None이면 표준 MBR(H=R).
        use_mecab: MeCab 형태소 분석
        metric: reference_keys가 None일 때 표준 mbr_ensemble에 전달
        multi_metrics: None이 아니면 ROUGE multi (rouge-1/2/l 평균)로 U 계산
        weights: {key: float} — 후보 key별 가중치 (없으면 1.0)
    """
    if not reference_keys:
        if multi_metrics:
            return mbr_multi_metric(candidates, use_mecab=use_mecab, metrics=multi_metrics)
        return mbr_ensemble(candidates, use_mecab=use_mecab, metric=metric)

    refs = [(k, t) for k, t in candidates if k in reference_keys]
    if not refs:
        if multi_metrics:
            return mbr_multi_metric(candidates, use_mecab=use_mecab, metrics=multi_metrics)
        return mbr_ensemble(candidates, use_mecab=use_mecab, metric=metric)

    try:
        from rouge import Rouge
    except ImportError:
        raise ImportError("rouge 패키지가 필요합니다: pip install rouge")

    rouge = Rouge()
    if weights is None:
        weights = {}

    if use_mecab:
        try:
            from .mecab_ko import get_mecab
            m = get_mecab()

            def to_morph(text):
                if text and text.strip():
                    morphs = " ".join(m.morphs(text))
                    return morphs if morphs else "빈요약"
                return "빈요약"
        except ImportError:
            def to_morph(text):
                return text if text and text.strip() else "빈요약"
    else:

        def to_morph(text):
            return text if text and text.strip() else "빈요약"

    ref_morphs = [(k, to_morph(t)) for k, t in refs]
    metrics = multi_metrics if multi_metrics else [metric]

    best_score = -1.0
    best_text = candidates[0][1]

    for ck, ctext in candidates:
        cm = to_morph(ctext)
        num = 0.0
        den = 0.0
        for rk, rm in ref_morphs:
            try:
                scores = rouge.get_scores([cm], [rm])[0]
                if multi_metrics:
                    u = sum(scores[m]["f"] for m in metrics) / len(metrics)
                else:
                    u = scores[metric]["f"]
                w = float(weights.get(rk, 1.0))
                num += u * w
                den += w
            except Exception:
                pass
        sc = num / den if den > 0 else 0.0
        if sc > best_score:
            best_score = sc
            best_text = ctext

    return best_text


def apply_mbr_to_dataset_asymmetric(
    test_df,
    all_predictions,
    reference_keys,
    use_mecab=True,
    metric="rouge-1",
    multi_metrics=None,
    weights=None,
    verbose=True,
):
    """
    apply_mbr_to_dataset과 동일하나 reference_keys로 비대칭 MBR.
    all_predictions: {key: [pred per row]}
    reference_keys: set of keys forming R (same length lists)
    """
    model_names = list(all_predictions.keys())
    n_samples = len(test_df) if hasattr(test_df, "__len__") else test_df

    mbr_preds = []
    model_selected = {name: 0 for name in model_names}

    iterator = range(n_samples)
    if verbose:
        iterator = tqdm(iterator, desc="MBR Ensemble (asymmetric)")

    for i in iterator:
        candidates = [(name, all_predictions[name][i]) for name in model_names]
        selected = mbr_ensemble_asymmetric(
            candidates,
            reference_keys,
            use_mecab=use_mecab,
            metric=metric,
            multi_metrics=multi_metrics,
            weights=weights,
        )
        mbr_preds.append(selected)
        for name, text in candidates:
            if text == selected:
                model_selected[name] += 1
                break

    if verbose:
        print("\n" + "=" * 80)
        print("MBR 앙상블 결과 - 모델 선택 빈도")
        print("=" * 80)
        for name, count in sorted(model_selected.items(), key=lambda x: -x[1]):
            percentage = 100 * count / n_samples
            bar = "█" * int(percentage / 2)
            print(f"  {name:40s}: {count:4d} ({percentage:5.1f}%) {bar}")
        print("=" * 80)

    return mbr_preds


def analyze_mbr_diversity(all_predictions, sample_idx=0, use_mecab=True):
    """
    MBR 후보들의 다양성 분석
    
    특정 샘플에 대해 각 프롬프트가 생성한 요약의 다양성을 분석합니다.
    
    Args:
        all_predictions: {prompt_name: [predictions]} 딕셔너리
        sample_idx: 분석할 샘플 인덱스
        use_mecab: Mecab 형태소 분석 사용 여부
    
    Returns:
        다양성 분석 결과 딕셔너리
    """
    try:
        from rouge import Rouge
    except ImportError:
        raise ImportError("rouge 패키지가 필요합니다: pip install rouge")
    
    rouge = Rouge()
    
    # 해당 샘플의 모든 후보 수집
    candidates = [(name, preds[sample_idx]) for name, preds in all_predictions.items()]
    
    # Mecab 형태소 분석
    if use_mecab:
        try:
            from .mecab_ko import get_mecab
            m = get_mecab()
            cand_morphs = [" ".join(m.morphs(c[1])) if c[1].strip() else "빈요약" 
                          for c in candidates]
        except ImportError:
            cand_morphs = [c[1] if c[1].strip() else "빈요약" for c in candidates]
    else:
        cand_morphs = [c[1] if c[1].strip() else "빈요약" for c in candidates]
    
    # Pairwise 유사도 행렬 계산
    n = len(candidates)
    similarity_matrix = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            try:
                scores = rouge.get_scores([cand_morphs[i]], [cand_morphs[j]])[0]
                pair = (candidates[i][0], candidates[j][0])
                similarity_matrix[pair] = {
                    "rouge-1": scores["rouge-1"]["f"],
                    "rouge-2": scores["rouge-2"]["f"],
                    "rouge-l": scores["rouge-l"]["f"],
                }
            except:
                pass
    
    # 통계 계산
    if similarity_matrix:
        rouge1_scores = [s["rouge-1"] for s in similarity_matrix.values()]
        avg_similarity = sum(rouge1_scores) / len(rouge1_scores)
        min_similarity = min(rouge1_scores)
        max_similarity = max(rouge1_scores)
    else:
        avg_similarity = min_similarity = max_similarity = 0
    
    return {
        "candidates": candidates,
        "similarity_matrix": similarity_matrix,
        "avg_similarity": avg_similarity,
        "min_similarity": min_similarity,
        "max_similarity": max_similarity,
        "diversity_score": 1 - avg_similarity,  # 다양성 = 1 - 평균 유사도
    }


if __name__ == "__main__":
    # 테스트 케이스
    print("=" * 80)
    print("MBR 디코딩 테스트")
    print("=" * 80)
    
    # 샘플 후보들
    test_candidates = [
        ("base", "#Person1#은 #Person2#에게 인사한다."),
        ("abstract", "#Person1#은 #Person2#와 인사를 나눈다."),
        ("oneshot", "#Person1#이 #Person2#에게 인사했다."),
        ("topic", "#Person1#과 #Person2#가 서로 인사합니다."),
        ("narrative", "#Person1#은 #Person2#에게 안녕이라고 말했다."),
    ]
    
    print("\n[후보 요약들]")
    for name, text in test_candidates:
        print(f"  {name:15s}: {text}")
    
    print("\n[MBR 선택 결과]")
    selected = mbr_ensemble(test_candidates, use_mecab=False)
    print(f"  선택된 요약: {selected}")
    
    print("\n" + "=" * 80)
    print("완료!")
    print("=" * 80)
