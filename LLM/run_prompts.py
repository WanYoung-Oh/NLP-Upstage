"""
프롬프트 엔지니어링 메인 실행 스크립트

학습, 추론, 평가를 위한 통합 인터페이스를 제공합니다.
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from prompts.inference import InferencePipeline, quick_inference
from prompts.evaluation import (
    evaluate_prompts,
    compare_base_vs_topic,
    evaluate_mbr_ensemble,
    generate_evaluation_report,
)
from prompts.mbr_prompts import get_all_prompt_variants


def _resolve_input_path(path_str: str) -> Path:
    """
    입력 경로를 실행 위치와 무관하게 해석합니다.

    우선순위:
    1) 절대경로/현재 작업 디렉터리 기준 존재
    2) 프로젝트 루트 기준 (NLP/)
    3) 프로젝트 루트/data 기준 (test.csv, dev.csv 단축 입력 지원)
    """
    p = Path(path_str).expanduser()
    if p.is_absolute() and p.exists():
        return p

    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    project_root = Path(__file__).resolve().parents[1]  # .../NLP
    root_candidate = (project_root / p).resolve()
    if root_candidate.exists():
        return root_candidate

    data_candidate = (project_root / "data" / p.name).resolve()
    if data_candidate.exists():
        return data_candidate

    raise FileNotFoundError(
        f"입력 파일을 찾을 수 없습니다: {path_str}\n"
        f"- 확인 경로: {cwd_candidate}, {root_candidate}, {data_candidate}"
    )


def _causal_lm_device_map():
    """
    bitsandbytes 4bit 로드 시 device_map='auto'가 VRAM 추정 오류 등으로
    일부를 CPU/디스크에 두면 ValueError가 납니다. 단일 GPU면 전부 cuda:0에 올립니다.
    """
    if torch.cuda.is_available():
        return {"": 0}
    return "auto"


def _resolve_output_path(path_str: str) -> Path:
    """
    출력 경로를 프로젝트 루트 기준으로 보정하고 상위 디렉토리를 생성합니다.
    """
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_model_and_tokenizer(model_path: str):
    """
    모델 경로가 full model인지 LoRA adapter인지 자동 판별하여 로드합니다.
    adapter인 경우 4-bit NF4 양자화로 베이스 모델을 로드합니다.
    """
    resolved_model_path = _resolve_input_path(model_path)

    tokenizer = AutoTokenizer.from_pretrained(str(resolved_model_path))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_config_path = resolved_model_path / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)

        base_model_name = adapter_cfg.get("base_model_name_or_path")
        if not base_model_name:
            raise ValueError(
                f"adapter_config.json에 base_model_name_or_path가 없습니다: {adapter_config_path}"
            )

        print(f"어댑터 체크포인트 감지: {resolved_model_path}")
        print(f"베이스 모델 로드 (4-bit NF4): {base_model_name}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, str(resolved_model_path))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(resolved_model_path),
            device_map=_causal_lm_device_map(),
            torch_dtype="auto",
            trust_remote_code=True,
        )

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="프롬프트 엔지니어링 실행 스크립트")
    prompt_variant_choices = list(get_all_prompt_variants().keys())
    
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
    parser.add_argument(
        "--prompt_variant",
        type=str,
        default="base",
        choices=prompt_variant_choices,
        help="비MBR 추론 시 사용할 단일 프롬프트 변형",
    )
    parser.add_argument("--variants_dir", type=str, default=None,
                       help="변형별 중간 결과 저장 디렉토리 (MBR 사용 시 각 변형 CSV 저장)")
    parser.add_argument("--use_topic", action="store_true",
                       help="Topic 정보 사용")
    parser.add_argument("--max_new_tokens", type=int, default=128,
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
    
    model, tokenizer = _load_model_and_tokenizer(args.model_path)
    
    print("✓ 모델 로드 완료")
    
    # 실행 모드에 따른 분기
    if args.mode == "inference":
        # 추론 모드
        print("\n" + "=" * 80)
        print("추론 모드")
        print("=" * 80)
        if args.use_mbr:
            selected_prompt_variants = None  # 전체 프롬프트 변형 사용 (MBR)
            print("실행 설정: MBR 앙상블 (전체 프롬프트 변형)")
        else:
            selected_prompt_variants = [args.prompt_variant]  # 단일 프롬프트 실행
            print(f"실행 설정: 단일 프롬프트 ({args.prompt_variant})")
        
        variants_dir = str(_resolve_output_path(args.variants_dir)) if args.variants_dir else None
        predictions = quick_inference(
            model=model,
            tokenizer=tokenizer,
            test_csv_path=str(_resolve_input_path(args.test_file)),
            output_csv_path=str(_resolve_output_path(args.output_file)),
            use_mbr=args.use_mbr,
            use_topic=args.use_topic,
            prompt_variants=selected_prompt_variants,
            variants_output_dir=variants_dir,
        )
        
        print(f"\n✓ 추론 완료: {len(predictions)}개 요약 생성")
        print(f"✓ 결과 저장: {args.output_file}")
    
    elif args.mode == "evaluate":
        # 평가 모드
        print("\n" + "=" * 80)
        print("평가 모드")
        print("=" * 80)
        
        dev_df = pd.read_csv(_resolve_input_path(args.dev_file))
        
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
        
        dev_df = pd.read_csv(_resolve_input_path(args.dev_file))
        
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
        
        dev_df = pd.read_csv(_resolve_input_path(args.dev_file))
        
        report = generate_evaluation_report(
            dev_df=dev_df,
            model=model,
            tokenizer=tokenizer,
            output_path=str(_resolve_output_path(args.report_path)),
        )
        
        print("\n✓ 보고서 생성 완료")
        print(f"✓ 보고서 저장: {args.report_path}")


if __name__ == "__main__":
    main()
