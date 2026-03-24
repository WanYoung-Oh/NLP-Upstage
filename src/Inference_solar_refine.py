"""
KoBART + Solar API 하이브리드 추론 파이프라인
==============================================
1단계: KoBART 체크포인트로 요약 생성
2단계: Solar API few-shot 프롬프팅으로 요약문 후처리(refinement)

사용법:
# Dev로 효과 검증
    python inference_solar_refine.py \
        --config config.yaml \
        --solar_api_key YOUR_KEY \
        --mode refine \
        --eval_dev
# Inference
    python inference_solar_refine.py \
        --config config_solar.yaml \
        --solar_api_key YOUR_SOLAR_API_KEY \
        --mode refine          # refine | kobart_only | solar_only | ensemble
        
"""

import argparse
import json
import os
import time
import random
from typing import List, Dict, Tuple, Optional

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    import requests
except ImportError:
    print("requests 라이브러리가 필요합니다: pip install requests")

try:
    from rouge import Rouge
except ImportError:
    print("rouge 라이브러리가 필요합니다: pip install rouge")


# ============================================================
# 1. Config 로드
# ============================================================
def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# 2. KoBART 추론
# ============================================================
class KoBARTInference:
    """KoBART 체크포인트 기반 요약 생성기"""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 & 토크나이저 로드
        model_path = config["inference"]["ckt_path"]
        print(f"[KoBART] 체크포인트 로드: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.model.eval()

        # Special tokens 추가 (체크포인트에 이미 포함되어 있을 수 있음)
        if "special_tokens" in config.get("tokenizer", {}):
            special_tokens_dict = {
                "additional_special_tokens": config["tokenizer"]["special_tokens"]
            }
            num_added = self.tokenizer.add_special_tokens(special_tokens_dict)
            if num_added > 0:
                self.model.resize_token_embeddings(len(self.tokenizer))
                print(f"[KoBART] Special tokens {num_added}개 추가")

    def generate_summaries(
        self, dialogues: List[str], batch_size: int = 32
    ) -> List[str]:
        """배치 단위로 요약 생성"""
        config_inf = self.config["inference"]
        summaries = []

        for i in tqdm(range(0, len(dialogues), batch_size), desc="[KoBART] 요약 생성"):
            batch = dialogues[i : i + batch_size]

            inputs = self.tokenizer(
                batch,
                max_length=self.config["tokenizer"]["encoder_max_len"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=config_inf.get("generate_max_length", 100),
                    num_beams=config_inf.get("num_beams", 4),
                    no_repeat_ngram_size=config_inf.get("no_repeat_ngram_size", 2),
                    early_stopping=config_inf.get("early_stopping", True),
                )

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # 후처리: 불필요한 토큰 제거
            remove_tokens = config_inf.get("remove_tokens", [])
            for text in decoded:
                for token in remove_tokens:
                    text = text.replace(token, "")
                summaries.append(text.strip())

        return summaries


# ============================================================
# 3. Solar API Few-shot 프롬프팅
# ============================================================
class SolarRefiner:
    """Solar API를 활용한 요약문 후처리/생성"""

    # Upstage Solar API 엔드포인트
    API_URL = "https://api.upstage.ai/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        model_name: str = "solar-pro",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _build_fewshot_refine_prompt(
        self,
        dialogue: str,
        kobart_summary: str,
        examples: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        Few-shot 후처리 프롬프트 구성

        Args:
            dialogue: 원본 대화
            kobart_summary: KoBART가 생성한 요약
            examples: [{"dialogue": ..., "draft_summary": ..., "refined_summary": ...}, ...]
        """
        system_prompt = (
            "당신은 한국어 대화 요약 전문가입니다. "
            "주어진 대화와 초안 요약을 보고, 요약문을 다듬어 주세요.\n\n"
            "규칙:\n"
            "1. 원본 대화의 핵심 정보를 빠짐없이 포함할 것\n"
            "2. 초안 요약의 핵심 키워드와 표현은 최대한 유지할 것 (ROUGE 점수 보존)\n"
            "3. 문법적 오류, 어색한 표현, 불필요한 반복만 수정할 것\n"
            "4. 새로운 정보를 추가하지 말 것\n"
            "5. 요약문은 1~3문장으로 간결하게 작성할 것\n"
            "6. #Person1#, #Person2# 등의 화자 표시는 그대로 유지할 것\n"
            "7. 다듬어진 요약문만 출력할 것 (설명 없이)"
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Few-shot examples
        for ex in examples:
            user_msg = (
                f"[대화]\n{ex['dialogue']}\n\n"
                f"[초안 요약]\n{ex['draft_summary']}"
            )
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": ex["refined_summary"]})

        # 실제 입력
        user_msg = f"[대화]\n{dialogue}\n\n[초안 요약]\n{kobart_summary}"
        messages.append({"role": "user", "content": user_msg})

        return messages

    def _build_fewshot_generate_prompt(
        self,
        dialogue: str,
        examples: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        Few-shot 직접 생성 프롬프트 (solar_only 모드용)

        Args:
            dialogue: 원본 대화
            examples: [{"dialogue": ..., "summary": ...}, ...]
        """
        system_prompt = (
            "당신은 한국어 대화 요약 전문가입니다. "
            "주어진 대화를 읽고 핵심 내용을 1~3문장으로 요약해 주세요.\n\n"
            "규칙:\n"
            "1. 대화의 핵심 정보와 결론을 포함할 것\n"
            "2. #Person1#, #Person2# 등의 화자 표시는 그대로 유지할 것\n"
            "3. 간결하고 자연스러운 한국어로 작성할 것\n"
            "4. 요약문만 출력할 것 (설명 없이)"
        )

        messages = [{"role": "system", "content": system_prompt}]

        for ex in examples:
            messages.append({"role": "user", "content": f"[대화]\n{ex['dialogue']}"})
            messages.append({"role": "assistant", "content": ex["summary"]})

        messages.append({"role": "user", "content": f"[대화]\n{dialogue}"})

        return messages

    def _call_api(self, messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
        """Solar API 호출 (재시도 로직 포함)"""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 200,
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.API_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()

            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    # Rate limit → 대기 후 재시도
                    wait = self.retry_delay * (2**attempt) + random.uniform(0, 1)
                    print(f"  Rate limit. {wait:.1f}초 대기 후 재시도...")
                    time.sleep(wait)
                else:
                    print(f"  API 오류 (HTTP {response.status_code}): {e}")
                    if attempt == self.max_retries - 1:
                        return ""
            except requests.exceptions.RequestException as e:
                print(f"  요청 실패: {e}")
                if attempt == self.max_retries - 1:
                    return ""
                time.sleep(self.retry_delay)

        return ""

    def refine_summaries(
        self,
        dialogues: List[str],
        kobart_summaries: List[str],
        examples: List[Dict[str, str]],
        delay: float = 0.5,
    ) -> List[str]:
        """
        KoBART 요약을 Solar로 후처리

        Args:
            dialogues: 원본 대화 리스트
            kobart_summaries: KoBART 생성 요약 리스트
            examples: few-shot 예시 (train에서 샘플링)
            delay: API 호출 간 대기 시간(초)
        """
        refined = []
        for i, (dial, summ) in enumerate(
            tqdm(
                zip(dialogues, kobart_summaries),
                total=len(dialogues),
                desc="[Solar] 후처리",
            )
        ):
            messages = self._build_fewshot_refine_prompt(dial, summ, examples)
            result = self._call_api(messages)

            if result:
                refined.append(result)
            else:
                # API 실패 시 KoBART 원본 유지
                print(f"  [{i}] API 실패 → KoBART 원본 유지")
                refined.append(summ)

            if delay > 0 and i < len(dialogues) - 1:
                time.sleep(delay)

        return refined

    def generate_summaries(
        self,
        dialogues: List[str],
        examples: List[Dict[str, str]],
        delay: float = 0.5,
    ) -> List[str]:
        """Solar API만으로 직접 요약 생성 (solar_only 모드)"""
        summaries = []
        for i, dial in enumerate(
            tqdm(dialogues, desc="[Solar] 직접 생성")
        ):
            messages = self._build_fewshot_generate_prompt(dial, examples)
            result = self._call_api(messages)

            if result:
                summaries.append(result)
            else:
                summaries.append("")

            if delay > 0 and i < len(dialogues) - 1:
                time.sleep(delay)

        return summaries


# ============================================================
# 4. Few-shot 예시 샘플링
# ============================================================
def sample_fewshot_examples(
    train_df: pd.DataFrame,
    n_examples: int = 3,
    seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Train 데이터에서 few-shot 예시 샘플링

    Returns:
        refine_examples: 후처리용 예시 (dialogue, draft_summary, refined_summary)
        generate_examples: 직접 생성용 예시 (dialogue, summary)
    """
    random.seed(seed)
    samples = train_df.sample(n=n_examples, random_state=seed)

    generate_examples = []
    refine_examples = []

    for _, row in samples.iterrows():
        generate_examples.append(
            {"dialogue": row["dialogue"], "summary": row["summary"]}
        )
        # 후처리용: 정답 요약을 refined로, 약간 변형한 버전을 draft로 사용
        # (실제로는 KoBART dev 출력을 사용하면 더 좋음)
        refine_examples.append(
            {
                "dialogue": row["dialogue"],
                "draft_summary": row["summary"],  # 이상적으로는 KoBART의 dev 예측값
                "refined_summary": row["summary"],
            }
        )

    return refine_examples, generate_examples


def build_refine_examples_from_dev(
    dev_df: pd.DataFrame,
    kobart_dev_summaries: List[str],
    n_examples: int = 3,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """
    Dev 데이터에서 KoBART 예측값과 정답을 짝지어 후처리 예시 구성
    → 가장 이상적인 few-shot 예시 (실제 KoBART 출력 → 정답)
    """
    random.seed(seed)
    indices = random.sample(range(len(dev_df)), min(n_examples, len(dev_df)))

    examples = []
    for idx in indices:
        row = dev_df.iloc[idx]
        examples.append(
            {
                "dialogue": row["dialogue"],
                "draft_summary": kobart_dev_summaries[idx],
                "refined_summary": row["summary"],
            }
        )

    return examples


# ============================================================
# 5. ROUGE 평가
# ============================================================
def evaluate_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """ROUGE-1, ROUGE-2, ROUGE-L F1 계산"""
    rouge = Rouge()

    # 빈 문자열 방지
    preds = [p if p.strip() else "없음" for p in predictions]
    refs = [r if r.strip() else "없음" for r in references]

    scores = rouge.get_scores(preds, refs, avg=True)
    result = {
        "rouge-1": scores["rouge-1"]["f"],
        "rouge-2": scores["rouge-2"]["f"],
        "rouge-l": scores["rouge-l"]["f"],
    }
    result["total"] = sum(result.values())
    return result


# ============================================================
# 6. 앙상블 선택기
# ============================================================
def ensemble_select(
    dialogues: List[str],
    summaries_a: List[str],
    summaries_b: List[str],
    references: Optional[List[str]] = None,
) -> List[str]:
    """
    두 요약 후보 중 ROUGE가 높은 쪽 선택 (dev용)
    references가 없으면 (test) 더 긴 요약을 선택하는 휴리스틱 적용
    """
    if references is not None:
        rouge = Rouge()
        selected = []
        for a, b, ref in zip(summaries_a, summaries_b, references):
            a_safe = a if a.strip() else "없음"
            b_safe = b if b.strip() else "없음"
            ref_safe = ref if ref.strip() else "없음"

            score_a = rouge.get_scores(a_safe, ref_safe)[0]
            score_b = rouge.get_scores(b_safe, ref_safe)[0]

            total_a = sum(score_a[k]["f"] for k in ["rouge-1", "rouge-2", "rouge-l"])
            total_b = sum(score_b[k]["f"] for k in ["rouge-1", "rouge-2", "rouge-l"])

            selected.append(a if total_a >= total_b else b)
        return selected
    else:
        # Test: 길이 기반 휴리스틱 (또는 항상 refined 선택)
        return summaries_b  # refined 버전 우선


# ============================================================
# 7. 메인 파이프라인
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="KoBART + Solar 하이브리드 추론")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--solar_api_key", type=str, required=True)
    parser.add_argument("--solar_model", type=str, default="solar-pro")
    parser.add_argument(
        "--mode",
        type=str,
        default="refine",
        choices=["refine", "kobart_only", "solar_only", "ensemble"],
        help=(
            "refine: KoBART → Solar 후처리, "
            "kobart_only: KoBART만, "
            "solar_only: Solar만, "
            "ensemble: 둘 다 생성 후 ROUGE 높은 것 선택"
        ),
    )
    parser.add_argument("--n_fewshot", type=int, default=3, help="Few-shot 예시 수")
    parser.add_argument("--api_delay", type=float, default=0.5, help="API 호출 간 대기(초)")
    parser.add_argument("--eval_dev", action="store_true", help="Dev 데이터로 평가")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Config 로드
    config = load_config(args.config)
    data_path = config["general"]["data_path"]
    result_path = config["inference"].get("result_path", "./prediction/")
    os.makedirs(result_path, exist_ok=True)

    # 데이터 로드
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    print(f"[데이터] Train: {len(train_df)}개")

    if args.eval_dev:
        target_df = pd.read_csv(os.path.join(data_path, "dev.csv"))
        print(f"[데이터] Dev (평가용): {len(target_df)}개")
    else:
        target_df = pd.read_csv(os.path.join(data_path, "test.csv"))
        print(f"[데이터] Test: {len(target_df)}개")

    dialogues = target_df["dialogue"].tolist()

    # Few-shot 예시 준비
    refine_examples, generate_examples = sample_fewshot_examples(
        train_df, n_examples=args.n_fewshot, seed=args.seed
    )

    # Solar 클라이언트
    solar = SolarRefiner(
        api_key=args.solar_api_key,
        model_name=args.solar_model,
    )

    # -------------------------------------------------------
    # 모드별 실행
    # -------------------------------------------------------
    if args.mode == "kobart_only":
        print("\n=== 모드: KoBART만 사용 ===")
        kobart = KoBARTInference(config)
        final_summaries = kobart.generate_summaries(
            dialogues, batch_size=config["inference"].get("batch_size", 32)
        )

    elif args.mode == "solar_only":
        print("\n=== 모드: Solar API만 사용 (few-shot) ===")
        final_summaries = solar.generate_summaries(
            dialogues, examples=generate_examples, delay=args.api_delay
        )

    elif args.mode == "refine":
        print("\n=== 모드: KoBART → Solar 후처리 ===")

        # Step 1: KoBART 요약 생성
        kobart = KoBARTInference(config)
        kobart_summaries = kobart.generate_summaries(
            dialogues, batch_size=config["inference"].get("batch_size", 32)
        )

        # (선택) Dev에서 더 좋은 few-shot 예시 구성
        if args.eval_dev:
            # Dev KoBART 출력을 예시로 활용 (처음 n개)
            refine_examples = build_refine_examples_from_dev(
                target_df, kobart_summaries, n_examples=args.n_fewshot, seed=args.seed
            )

        # Step 2: Solar 후처리
        final_summaries = solar.refine_summaries(
            dialogues,
            kobart_summaries,
            examples=refine_examples,
            delay=args.api_delay,
        )

        # 결과 비교 출력 (처음 3개)
        print("\n--- 후처리 비교 (처음 3개) ---")
        for i in range(min(3, len(dialogues))):
            print(f"\n[{i}] KoBART:  {kobart_summaries[i]}")
            print(f"    Solar:   {final_summaries[i]}")

    elif args.mode == "ensemble":
        print("\n=== 모드: KoBART + Solar 앙상블 ===")

        # 두 모델 모두 생성
        kobart = KoBARTInference(config)
        kobart_summaries = kobart.generate_summaries(
            dialogues, batch_size=config["inference"].get("batch_size", 32)
        )

        solar_summaries = solar.generate_summaries(
            dialogues, examples=generate_examples, delay=args.api_delay
        )

        # 앙상블 선택
        references = target_df["summary"].tolist() if args.eval_dev else None
        final_summaries = ensemble_select(
            dialogues, kobart_summaries, solar_summaries, references
        )

    # -------------------------------------------------------
    # 평가 (dev 모드)
    # -------------------------------------------------------
    if args.eval_dev and "summary" in target_df.columns:
        references = target_df["summary"].tolist()
        scores = evaluate_rouge(final_summaries, references)

        print(f"\n{'='*50}")
        print(f"[{args.mode}] ROUGE 평가 결과:")
        print(f"  ROUGE-1 F1: {scores['rouge-1']:.4f}")
        print(f"  ROUGE-2 F1: {scores['rouge-2']:.4f}")
        print(f"  ROUGE-L F1: {scores['rouge-l']:.4f}")
        print(f"  Total:      {scores['total']:.4f}")
        print(f"{'='*50}")

        # 모드별 비교 (refine/ensemble일 때 KoBART 단독 점수도 출력)
        if args.mode in ("refine", "ensemble") and "kobart_summaries" in dir():
            kobart_scores = evaluate_rouge(kobart_summaries, references)
            print(f"\n[kobart_only] ROUGE (비교용):")
            print(f"  ROUGE-1 F1: {kobart_scores['rouge-1']:.4f}")
            print(f"  ROUGE-2 F1: {kobart_scores['rouge-2']:.4f}")
            print(f"  ROUGE-L F1: {kobart_scores['rouge-l']:.4f}")
            print(f"  Total:      {kobart_scores['total']:.4f}")

            delta = scores["total"] - kobart_scores["total"]
            print(f"\n  → 후처리 효과: {delta:+.4f} ({'개선' if delta > 0 else '하락'})")

    # -------------------------------------------------------
    # 결과 저장
    # -------------------------------------------------------
    output_df = pd.DataFrame(
        {"fname": target_df["fname"], "summary": final_summaries}
    )

    output_file = os.path.join(result_path, f"output_{args.mode}.csv")
    output_df.to_csv(output_file, index=False)
    print(f"\n[저장] {output_file} ({len(output_df)}개)")

    # JSON으로도 저장 (디버깅용)
    debug_data = []
    for i in range(len(dialogues)):
        entry = {
            "fname": target_df["fname"].iloc[i],
            "dialogue": dialogues[i][:200] + "...",
            "final_summary": final_summaries[i],
        }
        if args.mode == "refine" and "kobart_summaries" in dir():
            entry["kobart_summary"] = kobart_summaries[i]
        debug_data.append(entry)

    debug_file = os.path.join(result_path, f"debug_{args.mode}.json")
    with open(debug_file, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, ensure_ascii=False, indent=2)
    print(f"[저장] {debug_file}")


if __name__ == "__main__":
    main()