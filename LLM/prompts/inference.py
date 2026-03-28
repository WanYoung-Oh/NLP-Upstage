"""
추론 파이프라인

8개 프롬프트로 추론 → MBR 앙상블 → 최종 요약 생성 파이프라인을 제공합니다.
학습된 모델을 사용하여 테스트 데이터에 대한 요약을 생성합니다.
"""

import os
import torch
from tqdm import tqdm
from typing import Optional, Dict, List
import pandas as pd

from .mbr_prompts import get_all_prompt_variants, create_messages
from .postprocess import postprocess_summary, advanced_postprocess
from .mbr_decoding import apply_mbr_to_dataset


class InferencePipeline:
    """
    추론 파이프라인 클래스
    
    모델 로드, 프롬프트 생성, 추론, 후처리, MBR 앙상블을 통합 관리합니다.
    
    Example:
        >>> pipeline = InferencePipeline(model, tokenizer)
        >>> results = pipeline.run(test_df, use_mbr=True)
    """
    
    def __init__(self, model, tokenizer, device="auto"):
        """
        Args:
            model: 학습된 모델
            tokenizer: 토크나이저
            device: 디바이스 ("cuda", "cpu", "auto")
        """
        self.model = model
        self.tokenizer = tokenizer
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # 모델을 디바이스로 이동
        if self.device != "auto":
            self.model = self.model.to(self.device)
        
        # 평가 모드
        self.model.eval()
    
    def generate_single(self, dialogue, topic="", prompt_variant="base", 
                       max_new_tokens=128, do_sample=False, **generation_kwargs):
        """
        단일 대화에 대한 요약 생성
        
        Args:
            dialogue: 대화 텍스트
            topic: 대화 주제 (선택)
            prompt_variant: 프롬프트 변형 이름
            max_new_tokens: 최대 생성 토큰 수
            do_sample: 샘플링 여부 (False면 Greedy)
            **generation_kwargs: 추가 생성 파라미터
        
        Returns:
            생성된 요약 텍스트
        """
        # 프롬프트 생성
        messages = create_messages(prompt_variant, dialogue, topic)
        
        # Chat Template 적용
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # 추론 시 True
            enable_thinking=False,        # Qwen3 Thinking 모드 비활성화
        )
        
        # 토크나이즈
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                **generation_kwargs
            )
        
        # 디코딩 (입력 부분 제외)
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        summary = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 후처리
        summary = postprocess_summary(summary)
        
        return summary
    
    def generate_with_prompts(self, df, prompt_variants=None, use_topic=False,
                             max_new_tokens=128, variants_output_dir=None, verbose=True):
        """
        여러 프롬프트로 전체 데이터셋 추론

        Args:
            df: 데이터프레임 (dialogue 컬럼 필수, topic 컬럼 선택)
            prompt_variants: 사용할 프롬프트 변형 리스트 (None이면 전체)
            use_topic: topic 프롬프트 사용 여부
            max_new_tokens: 최대 생성 토큰 수
            variants_output_dir: 변형별 중간 결과 저장 디렉토리 (None이면 저장 안 함)
            verbose: 진행 상황 출력 여부

        Returns:
            {prompt_name: [predictions]} 딕셔너리
        """
        # 프롬프트 변형 선택
        if prompt_variants is None:
            all_variants = get_all_prompt_variants()
            prompt_variants = list(all_variants.keys())

        if variants_output_dir:
            os.makedirs(variants_output_dir, exist_ok=True)

        all_predictions = {}

        for variant_name in prompt_variants:
            if verbose:
                print(f"\n{'='*80}")
                print(f"프롬프트 변형: {variant_name}")
                print(f"{'='*80}")

            predictions = []

            iterator = df.iterrows()
            if verbose:
                iterator = tqdm(list(df.iterrows()), desc=f"Generating ({variant_name})")

            for idx, row in iterator:
                dialogue = row['dialogue']
                topic = row.get('topic', '') if use_topic else ''

                # 요약 생성
                summary = self.generate_single(
                    dialogue=dialogue,
                    topic=topic,
                    prompt_variant=variant_name,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy 디코딩
                )

                predictions.append(summary)

            all_predictions[variant_name] = predictions

            if verbose:
                print(f"✓ {variant_name}: {len(predictions)}개 요약 생성 완료")

            # 변형 완료 즉시 저장
            if variants_output_dir:
                variant_file = os.path.join(variants_output_dir, f"{variant_name}.csv")
                pd.DataFrame({"fname": df["fname"], "summary": predictions}).to_csv(
                    variant_file, index=False
                )
                if verbose:
                    print(f"✓ 변형 저장: {variant_file}")

        return all_predictions
    
    def run(self, test_df, use_mbr=True, use_topic=False,
            prompt_variants=None, max_new_tokens=128,
            output_file=None, variants_output_dir=None, verbose=True):
        """
        전체 추론 파이프라인 실행
        
        Args:
            test_df: 테스트 데이터프레임
            use_mbr: MBR 앙상블 사용 여부
            use_topic: Topic 정보 사용 여부
            prompt_variants: 사용할 프롬프트 변형 리스트
            max_new_tokens: 최대 생성 토큰 수
            output_file: 결과 저장 파일 경로 (CSV)
            verbose: 진행 상황 출력 여부
        
        Returns:
            최종 요약 리스트
        """
        if verbose:
            print("=" * 80)
            print("추론 파이프라인 시작")
            print("=" * 80)
            print(f"데이터 크기: {len(test_df)}")
            print(f"MBR 앙상블: {'사용' if use_mbr else '미사용'}")
            print(f"Topic 정보: {'사용' if use_topic else '미사용'}")
            print(f"프롬프트 변형: {len(prompt_variants) if prompt_variants else '전체 (8개)'}")
            print("=" * 80)
        
        # 1. 여러 프롬프트로 추론 (변형 완료 즉시 저장 포함)
        all_predictions = self.generate_with_prompts(
            df=test_df,
            prompt_variants=prompt_variants,
            use_topic=use_topic,
            max_new_tokens=max_new_tokens,
            variants_output_dir=variants_output_dir,
            verbose=verbose,
        )

        # 2. MBR 앙상블 또는 단일 프롬프트 선택
        if use_mbr and len(all_predictions) > 1:
            if verbose:
                print("\n" + "=" * 80)
                print("MBR 앙상블 적용")
                print("=" * 80)
            
            final_predictions = apply_mbr_to_dataset(
                test_df=test_df,
                all_predictions=all_predictions,
                use_mecab=True,
                verbose=verbose,
            )
        else:
            # 첫 번째 프롬프트 결과 사용
            first_key = list(all_predictions.keys())[0]
            final_predictions = all_predictions[first_key]
            
            if verbose:
                print(f"\n단일 프롬프트 사용: {first_key}")
        
        # 3. 결과 저장 (선택)
        if output_file:
            result_df = test_df.copy()
            result_df['summary'] = final_predictions
            result_df.to_csv(output_file, index=False)
            
            if verbose:
                print(f"\n✓ 결과 저장: {output_file}")
        
        if verbose:
            print("\n" + "=" * 80)
            print("추론 파이프라인 완료")
            print("=" * 80)
        
        return final_predictions


def quick_inference(model, tokenizer, test_csv_path, output_csv_path,
                   use_mbr=False, use_topic=False, prompt_variants=None,
                   variants_output_dir=None):
    """
    빠른 추론 함수 (편의 함수)
    
    Args:
        model: 학습된 모델
        tokenizer: 토크나이저
        test_csv_path: 테스트 CSV 파일 경로
        output_csv_path: 출력 CSV 파일 경로
        use_mbr: MBR 앙상블 사용 여부
        use_topic: Topic 정보 사용 여부
        prompt_variants: 사용할 프롬프트 변형 리스트
    
    Returns:
        최종 요약 리스트
    
    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("path/to/model")
        >>> tokenizer = AutoTokenizer.from_pretrained("path/to/model")
        >>> results = quick_inference(
        ...     model, tokenizer,
        ...     "test.csv", "submission.csv",
        ...     use_mbr=True
        ... )
    """
    # 데이터 로드
    test_df = pd.read_csv(test_csv_path)
    
    # 파이프라인 실행
    pipeline = InferencePipeline(model, tokenizer)
    predictions = pipeline.run(
        test_df=test_df,
        use_mbr=use_mbr,
        use_topic=use_topic,
        prompt_variants=prompt_variants,
        output_file=output_csv_path,
        variants_output_dir=variants_output_dir,
        verbose=True,
    )
    
    return predictions


def batch_inference_with_dynamic_length(model, tokenizer, test_df, 
                                       batch_size=8, max_dialogue_length=2048):
    """
    배치 추론 (고급)
    
    대화 길이에 따라 동적으로 max_new_tokens를 조정하여 효율적으로 추론합니다.
    
    Args:
        model: 학습된 모델
        tokenizer: 토크나이저
        test_df: 테스트 데이터프레임
        batch_size: 배치 크기
        max_dialogue_length: 최대 대화 길이 (토큰)
    
    Returns:
        요약 리스트
    """
    # TODO: 배치 추론 구현
    # 현재는 단일 추론만 지원
    raise NotImplementedError("배치 추론은 아직 구현되지 않았습니다.")


if __name__ == "__main__":
    print("=" * 80)
    print("추론 파이프라인 사용 예시")
    print("=" * 80)
    
    example_code = '''
# 1. 모델 및 토크나이저 로드
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts.inference import InferencePipeline
import pandas as pd

model_path = "./outputs/checkpoint-best"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. 테스트 데이터 로드
test_df = pd.read_csv("test.csv")

# 3. 파이프라인 실행
pipeline = InferencePipeline(model, tokenizer)

# 옵션 1: MBR 앙상블 (8개 프롬프트)
predictions = pipeline.run(
    test_df=test_df,
    use_mbr=True,
    use_topic=False,
    output_file="submission.csv"
)

# 옵션 2: 단일 프롬프트 (빠른 추론)
predictions = pipeline.run(
    test_df=test_df,
    use_mbr=False,
    prompt_variants=["base"],
    output_file="submission_fast.csv"
)

# 옵션 3: Topic 정보 활용
predictions = pipeline.run(
    test_df=test_df,
    use_mbr=True,
    use_topic=True,
    output_file="submission_topic.csv"
)
    '''
    
    print(example_code)
    
    print("\n" + "=" * 80)
    print("예상 추론 시간 (RTX 3090 기준)")
    print("=" * 80)
    print("  단일 프롬프트 (Greedy):  ~5분")
    print("  MBR 4개 프롬프트:        ~20분")
    print("  MBR 8개 프롬프트:        ~40분")
    print("  MBR 16개 프롬프트:       ~80분")
    print("=" * 80)
