"""
프롬프트 엔지니어링 메인 실행 스크립트

학습, 추론, 평가를 위한 통합 인터페이스를 제공합니다.
"""

import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts.inference import InferencePipeline, quick_inference
from prompts.evaluation import (
    evaluate_prompts,
    compare_base_vs_topic,
    evaluate_mbr_ensemble,
    generate_evaluation_report,
)


def main():
    parser = argparse.ArgumentParser(description="프롬프트 엔지니어링 실행 스크립트")
    
    # 공통 인자
    parser.add_argument("--model_path", type=str, required=True,
                       help="학습된 모델 경로")
    parser.add_argument("--mode", type=str, required=True,
                       choices=["inference", "evaluate", "compare", "report"],
                       help="실행 모드")
    
    # 추론 관련 인자
    parser.add_argument("--test_file", type=str, default="test.csv",
                       help="테스트 CSV 파일 경로")
    parser.add_argument("--output_file", type=str, default="submission.csv",
                       help="출력 CSV 파일 경로")
    parser.add_argument("--use_mbr", action="store_true",
                       help="MBR 앙상블 사용")
    parser.add_argument("--use_topic", action="store_true",
                       help="Topic 정보 사용")
    parser.add_argument("--max_new_tokens", type=int, default=192,
                       help="최대 생성 토큰 수")
    
    # 평가 관련 인자
    parser.add_argument("--dev_file", type=str, default="dev.csv",
                       help="Dev CSV 파일 경로")
    parser.add_argument("--report_path", type=str, default="evaluation_report.txt",
                       help="평가 보고서 저장 경로")
    
    args = parser.parse_args()
    
    # 모델 및 토크나이저 로드
    print("=" * 80)
    print("모델 로딩 중...")
    print("=" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype="auto",
    )
    
    print("✓ 모델 로드 완료")
    
    # 실행 모드에 따른 분기
    if args.mode == "inference":
        # 추론 모드
        print("\n" + "=" * 80)
        print("추론 모드")
        print("=" * 80)
        
        predictions = quick_inference(
            model=model,
            tokenizer=tokenizer,
            test_csv_path=args.test_file,
            output_csv_path=args.output_file,
            use_mbr=args.use_mbr,
            use_topic=args.use_topic,
        )
        
        print(f"\n✓ 추론 완료: {len(predictions)}개 요약 생성")
        print(f"✓ 결과 저장: {args.output_file}")
    
    elif args.mode == "evaluate":
        # 평가 모드
        print("\n" + "=" * 80)
        print("평가 모드")
        print("=" * 80)
        
        dev_df = pd.read_csv(args.dev_file)
        
        results = evaluate_prompts(
            dev_df=dev_df,
            model=model,
            tokenizer=tokenizer,
            verbose=True,
        )
        
        print("\n✓ 평가 완료")
    
    elif args.mode == "compare":
        # Base vs Topic 비교 모드
        print("\n" + "=" * 80)
        print("Base vs Topic 비교 모드")
        print("=" * 80)
        
        dev_df = pd.read_csv(args.dev_file)
        
        comparison = compare_base_vs_topic(
            dev_df=dev_df,
            model=model,
            tokenizer=tokenizer,
            verbose=True,
        )
        
        print("\n✓ 비교 완료")
    
    elif args.mode == "report":
        # 종합 보고서 생성 모드
        print("\n" + "=" * 80)
        print("종합 보고서 생성 모드")
        print("=" * 80)
        
        dev_df = pd.read_csv(args.dev_file)
        
        report = generate_evaluation_report(
            dev_df=dev_df,
            model=model,
            tokenizer=tokenizer,
            output_path=args.report_path,
        )
        
        print("\n✓ 보고서 생성 완료")
        print(f"✓ 보고서 저장: {args.report_path}")


if __name__ == "__main__":
    main()
