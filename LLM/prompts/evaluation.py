"""
평가 스크립트

각 프롬프트 변형별 ROUGE 점수를 비교하고 성능을 분석합니다.
MeCab 형태소 분석 기반 ROUGE-1/2/L F1 점수를 계산합니다.
"""

import pandas as pd
from typing import Dict, List, Optional
from tqdm import tqdm


def evaluate_rouge(predictions, references, use_mecab=True):
    """
    ROUGE 점수 계산
    
    Args:
        predictions: 예측 요약 리스트
        references: 정답 요약 리스트
        use_mecab: MeCab 형태소 분석 사용 여부
    
    Returns:
        ROUGE 점수 딕셔너리
    
    Example:
        >>> preds = ["#Person1#은 #Person2#에게 인사한다."]
        >>> refs = ["#Person1#이 #Person2#에게 인사했다."]
        >>> scores = evaluate_rouge(preds, refs, use_mecab=True)
        >>> print(scores['rouge-1'])
    """
    try:
        from rouge import Rouge
    except ImportError:
        raise ImportError("rouge 패키지가 필요합니다: pip install rouge")
    
    rouge = Rouge()
    
    # MeCab 형태소 분석 (대회 평가 기준)
    if use_mecab:
        try:
            from .mecab_ko import get_mecab
            m = get_mecab()
            
            preds_m = []
            refs_m = []
            
            for p, r in zip(predictions, references):
                # 빈 텍스트 처리
                p_morphs = " ".join(m.morphs(p)) if p.strip() else "빈요약"
                r_morphs = " ".join(m.morphs(r)) if r.strip() else "빈요약"
                
                preds_m.append(p_morphs)
                refs_m.append(r_morphs)
        
        except ImportError:
            print("Warning: MeCab을 사용할 수 없습니다. 원본 텍스트를 사용합니다.")
            preds_m = [p if p.strip() else "빈요약" for p in predictions]
            refs_m = [r if r.strip() else "빈요약" for r in references]
    else:
        preds_m = [p if p.strip() else "빈요약" for p in predictions]
        refs_m = [r if r.strip() else "빈요약" for r in references]
    
    # ROUGE 점수 계산
    scores = rouge.get_scores(preds_m, refs_m, avg=True)
    
    return {
        'rouge-1': scores['rouge-1']['f'],
        'rouge-2': scores['rouge-2']['f'],
        'rouge-l': scores['rouge-l']['f'],
        'total': scores['rouge-1']['f'] + scores['rouge-2']['f'] + scores['rouge-l']['f'],
    }


def evaluate_prompts(dev_df, model, tokenizer, prompt_variants=None, 
                    use_topic=False, max_new_tokens=128, verbose=True):
    """
    각 프롬프트 변형의 성능 평가
    
    Args:
        dev_df: Dev 데이터프레임 (summary 컬럼 필수)
        model: 학습된 모델
        tokenizer: 토크나이저
        prompt_variants: 평가할 프롬프트 변형 리스트 (None이면 전체)
        use_topic: Topic 정보 사용 여부
        max_new_tokens: 최대 생성 토큰 수
        verbose: 진행 상황 출력 여부
    
    Returns:
        성능 비교 DataFrame
    
    Example:
        >>> results = evaluate_prompts(dev_df, model, tokenizer)
        >>> print(results.sort_values('rouge-1', ascending=False))
    """
    from .inference import InferencePipeline
    from .mbr_prompts import get_all_prompt_variants
    
    # 프롬프트 변형 선택
    if prompt_variants is None:
        all_variants = get_all_prompt_variants()
        prompt_variants = list(all_variants.keys())
    
    # 파이프라인 생성
    pipeline = InferencePipeline(model, tokenizer)
    
    results = {}
    
    for variant_name in prompt_variants:
        if verbose:
            print(f"\n{'='*80}")
            print(f"평가 중: {variant_name}")
            print(f"{'='*80}")
        
        predictions = []
        
        iterator = dev_df.iterrows()
        if verbose:
            iterator = tqdm(list(dev_df.iterrows()), desc=f"Evaluating ({variant_name})")
        
        for idx, row in iterator:
            dialogue = row['dialogue']
            topic = row.get('topic', '') if use_topic else ''
            
            # 요약 생성
            summary = pipeline.generate_single(
                dialogue=dialogue,
                topic=topic,
                prompt_variant=variant_name,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            
            predictions.append(summary)
        
        # ROUGE 평가
        scores = evaluate_rouge(predictions, dev_df['summary'].tolist(), use_mecab=True)
        results[variant_name] = scores
        
        if verbose:
            print(f"✓ ROUGE-1: {scores['rouge-1']:.4f}, "
                  f"ROUGE-2: {scores['rouge-2']:.4f}, "
                  f"ROUGE-L: {scores['rouge-l']:.4f}, "
                  f"Total: {scores['total']:.4f}")
    
    # DataFrame으로 변환
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('rouge-1', ascending=False)
    
    if verbose:
        print("\n" + "="*80)
        print("프롬프트 변형별 성능 비교")
        print("="*80)
        print(results_df.to_string())
        print("="*80)
    
    return results_df


def compare_base_vs_topic(dev_df, model, tokenizer, max_new_tokens=128, verbose=True):
    """
    Base 프롬프트 vs Topic 프롬프트 성능 비교
    
    Args:
        dev_df: Dev 데이터프레임
        model: 학습된 모델
        tokenizer: 토크나이저
        max_new_tokens: 최대 생성 토큰 수
        verbose: 진행 상황 출력 여부
    
    Returns:
        비교 결과 DataFrame
    """
    from .inference import InferencePipeline
    
    pipeline = InferencePipeline(model, tokenizer)
    
    results = {}
    
    # Base 프롬프트 평가
    if verbose:
        print("\n" + "="*80)
        print("Base 프롬프트 평가")
        print("="*80)
    
    base_predictions = []
    for idx, row in tqdm(list(dev_df.iterrows()), desc="Base 평가"):
        summary = pipeline.generate_single(
            dialogue=row['dialogue'],
            topic='',
            prompt_variant='base',
            max_new_tokens=max_new_tokens,
        )
        base_predictions.append(summary)
    
    base_scores = evaluate_rouge(base_predictions, dev_df['summary'].tolist())
    results['base'] = base_scores
    
    if verbose:
        print(f"✓ ROUGE-1: {base_scores['rouge-1']:.4f}")
    
    # Topic 프롬프트 평가
    if verbose:
        print("\n" + "="*80)
        print("Topic 프롬프트 평가")
        print("="*80)
    
    topic_predictions = []
    for idx, row in tqdm(list(dev_df.iterrows()), desc="Topic 평가"):
        summary = pipeline.generate_single(
            dialogue=row['dialogue'],
            topic=row.get('topic', ''),
            prompt_variant='topic',
            max_new_tokens=max_new_tokens,
        )
        topic_predictions.append(summary)
    
    topic_scores = evaluate_rouge(topic_predictions, dev_df['summary'].tolist())
    results['topic'] = topic_scores
    
    if verbose:
        print(f"✓ ROUGE-1: {topic_scores['rouge-1']:.4f}")
    
    # 비교 결과
    comparison_df = pd.DataFrame(results).T
    
    if verbose:
        print("\n" + "="*80)
        print("Base vs Topic 비교")
        print("="*80)
        print(comparison_df.to_string())
        
        # 개선 효과 계산
        improvement = topic_scores['rouge-1'] - base_scores['rouge-1']
        print(f"\nTopic 프롬프트 개선: {improvement:+.4f} ({improvement/base_scores['rouge-1']*100:+.2f}%)")
        print("="*80)
    
    return comparison_df


def evaluate_mbr_ensemble(dev_df, model, tokenizer, prompt_variants=None, 
                         use_topic=False, max_new_tokens=128, verbose=True):
    """
    MBR 앙상블 성능 평가
    
    Args:
        dev_df: Dev 데이터프레임
        model: 학습된 모델
        tokenizer: 토크나이저
        prompt_variants: 사용할 프롬프트 변형 리스트
        use_topic: Topic 정보 사용 여부
        max_new_tokens: 최대 생성 토큰 수
        verbose: 진행 상황 출력 여부
    
    Returns:
        평가 결과 딕셔너리
    """
    from .inference import InferencePipeline
    
    pipeline = InferencePipeline(model, tokenizer)
    
    if verbose:
        print("\n" + "="*80)
        print("MBR 앙상블 평가")
        print("="*80)
    
    # MBR 앙상블로 추론
    final_predictions = pipeline.run(
        test_df=dev_df,
        use_mbr=True,
        use_topic=use_topic,
        prompt_variants=prompt_variants,
        max_new_tokens=max_new_tokens,
        verbose=verbose,
    )
    
    # ROUGE 평가
    scores = evaluate_rouge(final_predictions, dev_df['summary'].tolist())
    
    if verbose:
        print("\n" + "="*80)
        print("MBR 앙상블 결과")
        print("="*80)
        print(f"ROUGE-1: {scores['rouge-1']:.4f}")
        print(f"ROUGE-2: {scores['rouge-2']:.4f}")
        print(f"ROUGE-L: {scores['rouge-l']:.4f}")
        print(f"Total:   {scores['total']:.4f}")
        print("="*80)
    
    return scores


def generate_evaluation_report(dev_df, model, tokenizer, output_path="evaluation_report.txt"):
    """
    종합 평가 보고서 생성
    
    다양한 프롬프트 변형과 MBR 앙상블의 성능을 종합적으로 평가합니다.
    
    Args:
        dev_df: Dev 데이터프레임
        model: 학습된 모델
        tokenizer: 토크나이저
        output_path: 보고서 저장 경로
    """
    report = []
    report.append("="*80)
    report.append("프롬프트 엔지니어링 평가 보고서")
    report.append("="*80)
    report.append("")
    
    # 1. Base vs Topic 비교
    report.append("1. Base vs Topic 프롬프트 비교")
    report.append("-"*80)
    comparison = compare_base_vs_topic(dev_df, model, tokenizer, verbose=False)
    report.append(comparison.to_string())
    report.append("")
    
    # 2. 모든 프롬프트 변형 평가
    report.append("2. 모든 프롬프트 변형 성능")
    report.append("-"*80)
    all_results = evaluate_prompts(dev_df, model, tokenizer, verbose=False)
    report.append(all_results.to_string())
    report.append("")
    
    # 3. MBR 앙상블 평가
    report.append("3. MBR 앙상블 성능")
    report.append("-"*80)
    mbr_scores = evaluate_mbr_ensemble(dev_df, model, tokenizer, verbose=False)
    report.append(f"ROUGE-1: {mbr_scores['rouge-1']:.4f}")
    report.append(f"ROUGE-2: {mbr_scores['rouge-2']:.4f}")
    report.append(f"ROUGE-L: {mbr_scores['rouge-l']:.4f}")
    report.append(f"Total:   {mbr_scores['total']:.4f}")
    report.append("")
    
    report.append("="*80)
    
    # 보고서 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"✓ 평가 보고서 저장: {output_path}")
    
    return '\n'.join(report)


if __name__ == "__main__":
    print("=" * 80)
    print("평가 스크립트 사용 예시")
    print("=" * 80)
    
    example_code = '''
# 1. 모델 및 데이터 로드
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts.evaluation import (
    evaluate_prompts,
    compare_base_vs_topic,
    evaluate_mbr_ensemble,
    generate_evaluation_report,
)
import pandas as pd

model_path = "./outputs/checkpoint-best"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

dev_df = pd.read_csv("dev.csv")

# 2. Base vs Topic 비교
comparison = compare_base_vs_topic(dev_df, model, tokenizer)

# 3. 모든 프롬프트 변형 평가
results = evaluate_prompts(dev_df, model, tokenizer)
print(results.sort_values('rouge-1', ascending=False))

# 4. MBR 앙상블 평가
mbr_scores = evaluate_mbr_ensemble(dev_df, model, tokenizer)

# 5. 종합 보고서 생성
report = generate_evaluation_report(dev_df, model, tokenizer)
    '''
    
    print(example_code)
    
    print("\n" + "=" * 80)
    print("예상 평가 시간 (dev.csv 499개 샘플 기준)")
    print("=" * 80)
    print("  단일 프롬프트 평가:     ~2-3분")
    print("  8개 프롬프트 평가:      ~20-25분")
    print("  MBR 앙상블 평가:        ~40-45분")
    print("  종합 보고서 생성:       ~60-70분")
    print("=" * 80)
