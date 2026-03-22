"""
간단한 사용 예시 스크립트

프롬프트 엔지니어링 모듈을 사용하여 빠르게 추론하는 예시입니다.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from prompts.inference import InferencePipeline
import pandas as pd
import torch
from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../NLP


def _data_file(name: str) -> Path:
    """data 디렉토리 기준 파일 경로를 반환합니다."""
    return PROJECT_ROOT / "data" / name


def _resolve_model_path(model_path: str) -> Path:
    """모델 경로를 절대 경로로 변환합니다."""
    p = Path(model_path).expanduser()
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def _load_model_and_tokenizer(model_path: str):
    """
    모델 경로가 full model인지 LoRA adapter인지 자동 판별하여 로드합니다.
    """
    resolved_model_path = _resolve_model_path(model_path)

    tokenizer = AutoTokenizer.from_pretrained(str(resolved_model_path))
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
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
        print(f"베이스 모델 로드: {base_model_name}")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, str(resolved_model_path))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(resolved_model_path),
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True,
        )

    return model, tokenizer


def example_1_single_prompt():
    """예시 1: 단일 프롬프트로 빠른 추론"""
    print("=" * 80)
    print("예시 1: 단일 프롬프트 추론 (Greedy)")
    print("=" * 80)
    
    # 모델 로드 (실제 경로로 변경 필요)
    model_path = "./outputs/checkpoint-best"  # 실제 모델 경로
    
    print("모델 로딩 중...")
    model, tokenizer = _load_model_and_tokenizer(model_path)
    
    # 테스트 데이터 로드
    test_df = pd.read_csv(_data_file("test.csv"))
    print(f"테스트 데이터 크기: {len(test_df)}")
    
    # 파이프라인 생성
    pipeline = InferencePipeline(model, tokenizer)
    
    # 단일 프롬프트로 추론
    predictions = pipeline.run(
        test_df=test_df,
        use_mbr=False,  # MBR 사용 안함
        prompt_variants=["base"],  # Base 프롬프트만 사용
        output_file="submission_fast.csv",
    )
    
    print(f"\n✓ 추론 완료: {len(predictions)}개 요약 생성")
    print(f"✓ 첫 번째 요약 예시: {predictions[0]}")


def example_2_mbr_ensemble():
    """예시 2: MBR 앙상블로 최고 성능 추론"""
    print("\n" + "=" * 80)
    print("예시 2: MBR 앙상블 추론 (8개 프롬프트)")
    print("=" * 80)
    
    # 모델 로드
    model_path = "./outputs/checkpoint-best"
    
    print("모델 로딩 중...")
    model, tokenizer = _load_model_and_tokenizer(model_path)
    
    # 테스트 데이터 로드
    test_df = pd.read_csv(_data_file("test.csv"))
    
    # 파이프라인 생성
    pipeline = InferencePipeline(model, tokenizer)
    
    # MBR 앙상블로 추론
    predictions = pipeline.run(
        test_df=test_df,
        use_mbr=True,  # MBR 사용
        use_topic=False,
        output_file="submission_mbr.csv",
    )
    
    print(f"\n✓ MBR 앙상블 완료: {len(predictions)}개 요약 생성")
    print("✓ 예상 ROUGE-1: 0.570-0.575")


def example_3_evaluate():
    """예시 3: Dev 세트 평가"""
    print("\n" + "=" * 80)
    print("예시 3: Dev 세트 평가")
    print("=" * 80)
    
    from prompts.evaluation import compare_base_vs_topic
    
    # 모델 로드
    model_path = "./outputs/checkpoint-best"
    
    print("모델 로딩 중...")
    model, tokenizer = _load_model_and_tokenizer(model_path)
    
    # Dev 데이터 로드
    dev_df = pd.read_csv(_data_file("dev.csv"))
    print(f"Dev 데이터 크기: {len(dev_df)}")
    
    # Base vs Topic 비교
    comparison = compare_base_vs_topic(
        dev_df=dev_df,
        model=model,
        tokenizer=tokenizer,
        verbose=True,
    )
    
    print("\n✓ 평가 완료")
    print(comparison)


def example_4_single_dialogue():
    """예시 4: 단일 대화 요약 생성"""
    print("\n" + "=" * 80)
    print("예시 4: 단일 대화 요약")
    print("=" * 80)
    
    # 모델 로드
    model_path = "./outputs/checkpoint-best"
    
    print("모델 로딩 중...")
    model, tokenizer = _load_model_and_tokenizer(model_path)
    
    # 파이프라인 생성
    pipeline = InferencePipeline(model, tokenizer)
    
    # 샘플 대화
    sample_dialogue = """#Person1#: 안녕하세요, Mr. Smith. 저는 Dr. Hawkins입니다. 오늘 무슨 일로 오셨어요? 
#Person2#: 건강검진을 받으려고 왔어요. 
#Person1#: 네, 5년 동안 검진을 안 받으셨네요. 매년 한 번씩 받으셔야 해요. 
#Person2#: 알죠. 특별히 아픈 데가 없으면 굳이 갈 필요가 없다고 생각했어요."""
    
    # 요약 생성
    summary = pipeline.generate_single(
        dialogue=sample_dialogue,
        topic="건강검진",
        prompt_variant="base",
    )
    
    print("\n[대화]")
    print(sample_dialogue)
    print("\n[요약]")
    print(summary)


if __name__ == "__main__":
    print("프롬프트 엔지니어링 사용 예시")
    print("=" * 80)
    print("실행하려면 주석을 해제하세요:")
    print("1. example_1_single_prompt()      # 단일 프롬프트 추론")
    print("2. example_2_mbr_ensemble()       # MBR 앙상블 추론")
    print("3. example_3_evaluate()           # Dev 세트 평가")
    print("4. example_4_single_dialogue()    # 단일 대화 요약")
    print("=" * 80)
    
    # 실행할 예시를 주석 해제하세요
    # example_1_single_prompt()
    # example_2_mbr_ensemble()
    # example_3_evaluate()
    # example_4_single_dialogue()
    
    print("\n✓ 스크립트를 참고하여 원하는 방식으로 사용하세요!")
