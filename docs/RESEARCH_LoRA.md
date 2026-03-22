# RESEARCH.md — 한국어 다화자 대화 요약 프로젝트 (최종 업데이트)

> **최종 수정일**: 2026년 3월 20일  
> **환경**: NVIDIA RTX 3090 (24GB VRAM)  
> **과제**: 다화자 대화(`#Person1#`, `#Person2#`, ...) → 한국어 요약 생성  
> **평가 지표**: Mecab 형태소 기반 ROUGE-1/2/L F1

---

## 목차

1. [데이터셋 분석](#1-데이터셋-분석)  
2. [베이스라인 코드 분석](#2-베이스라인-코드-분석)  
3. [모델 후보군 상세 분석](#3-모델-후보군-상세-분석)  
4. [SFT 전략](#4-sft-전략)  
5. [SimPO 전략](#5-simpo-전략)  
6. [VRAM 최적화 및 메모리 예산](#6-vram-최적화-및-메모리-예산)  
7. [디코딩 전략 (MBR 앙상블)](#7-디코딩-전략-mbr-앙상블)  
8. [데이터 증강 분석](#8-데이터-증강-분석)  
9. [프롬프트 엔지니어링 전략](#9-프롬프트-엔지니어링-전략)  
10. [평가 파이프라인](#10-평가-파이프라인)  
11. [실험 로드맵 및 우선순위](#11-실험-로드맵-및-우선순위)  
12. [참고문헌 및 출처](#12-참고문헌-및-출처)

---

## 최종 운영 전략 요약

본 프로젝트는 RTX 3090 (24GB VRAM) 환경에서 한국어 다화자 대화 요약 모델을 개발합니다. train.csv 분석 결과 일상적인 대화 위주로 구성되어 있어 고차원 논리 추론보다 직관적인 요약 능력에 집중합니다.

### 1. 모델 후보군 및 활용 전략

**Primary Model: Qwen3-14B**
- 활용 방식: 4-bit QLoRA + Response-Only SFT + SimPO
- 최적화 설정: `enable_thinking=False` (일상 대화에 불필요한 추론 과정 제거, VRAM 절약)
- LoRA: Rank 32로 상향하여 한국어 대화체 학습 밀도 강화
- 현재 베이스라인 코드의 검증된 모델로, RTX 3090에서 안정적 학습 가능

**Teacher Model: Llama 4 Maverick (API 전용)**
- 역할: 데이터 증강 및 검증용 교사 모델
- 현실적 제약: RTX 3090 단일 환경에서 직접 SFT/SimPO 학습 불가능
- 활용 전략: API를 통해 train.csv 데이터를 정제하거나, SimPO 학습을 위한 선호도(Chosen/Rejected) 데이터 생성

### 2. 하드웨어 및 VRAM 최적화

**Attention 최적화: Flash Attention 2**
- RTX 3090은 Ampere 아키텍처로 Flash Attention 3의 하드웨어 가속을 지원하지 않음
- FA2 + Unsloth 커널 최적화 조합으로 메모리 효율 극대화

**메모리 절약 기술**
- 8-bit Optimizer (`adamw_8bit`) 및 Gradient Checkpointing 활성화
- `max_seq_length=2048` 설정으로 긴 대화 로그 처리 시 트런케이션 방지

### 3. 학습 및 평가 파이프라인

**SFT (Supervised Fine-Tuning)**
- Response-Only Loss: 프롬프트 부분을 제외하고 어시스턴트 응답(summary) 구간만 Loss 계산
- 하이퍼파라미터: LR 2e-4, Epochs 3, Effective Batch Size 32 (1×32)

**SimPO (Simple Preference Optimization)**
- 목적: 요약문의 간결성과 사실 관계 정확도 향상
- Reference Model 불필요로 24GB VRAM 내에서 효율적 정렬(Alignment) 수행

**평가 전략**
- **ROUGE Score**: 대회 공식 평가지표인 ROUGE-1/2/L 점수 기반 모델 체크포인트 선별

### 4. 실험 로드맵 및 우선순위

**우선순위 1**: Qwen3-14B (R=32) 기반 Response-Only SFT 베이스라인 구축

**우선순위 2**: SFT 모델 결과물에 대해 Llama 4 API를 활용한 사실 관계 검증 및 SimPO 데이터셋 구성

**우선순위 3**: SimPO 정렬 학습을 통해 요약문의 품질 고도화 및 최종 제출 파일 생성

> **핵심 포인트**: Flash Attention 2, LoRA R=32 상향, Thinking 모드 제외는 RTX 3090의 아키텍처 한계와 train.csv 데이터 특성을 가장 실전적으로 반영한 설정으로, 성능과 안정성을 동시에 확보할 수 있습니다.

---

## 1. 데이터셋 분석

### 1-1. 데이터 구성 개요

| 분할 | 샘플 수 | 비고 |
|------|--------|------|
| `train.csv` | **17,341** | 원본 + 백트랜스레이션 증강(약 2,500개 포함) |
| `dev.csv` | **499** | 원본 검증 데이터 |
| `test.csv` | **499** | Summary 없음, 제출용 |

**컬럼 구성**

- `fname`: 파일명 ID (예: `train_0`)
- `dialogue`: 화자 태그 포함 멀티턴 대화 (`#Person1#:`, `#Person2#:`, ...)
- `summary`: 골드 요약문 (train/dev 전용)
- `topic`: 대화 주제 키워드 (train/dev 전용)

### 1-2. 통계적 특성

**대화(dialogue) 길이 분포**

| 분할 | 최소 | 최대 | 평균 | 중앙값 |
|------|-----|-----|-----|------|
| train | 80자 | 2,142자 | 399.8자 | 364.0자 |
| dev | 114자 | 1,269자 | 400.1자 | 367.0자 |
| test | 111자 | 2,275자 | 422.1자 | 386.0자 |

**요약(summary) 길이 분포 (train/dev)**

| 분할 | 최소 | 최대 | 평균 |
|------|-----|-----|-----|
| train | 13자 | 376자 | 85.8자 |
| dev | 29자 | 283자 | 81.2자 |

**화자 태그 사용 빈도 (train)**

| 태그 | 등장 횟수 | 비율 |
|-----|---------|-----|
| `#Person1#` | 17,341 | 100% (전체) |
| `#Person2#` | 17,341 | 100% (전체) |
| `#Person3#` | 175 | ~1.0% |
| `#Person4#` | 27 | ~0.16% |
| `#Person5#` 이상 | 10 | ~0.06% |

**주제(topic) 다양성**: train 기준 9,235개 고유 토픽 → 매우 다양한 도메인 커버

### 1-3. 핵심 관찰 및 설계 시사점

**대화 구조 특성**
- 대화의 99%는 2인 대화(`#Person1#` + `#Person2#`), 나머지는 3~7인 구조
- 평균 대화 길이 ~400자는 한국어 토큰 기준 약 150~250 토큰 수준 (BPE)
- 최대 2,275자는 약 700~900 토큰 → `max_seq_length=2048` 설정 시 전체 입력(프롬프트 + 대화 + 요약) 충분히 수용 가능

**요약 특성**
- 요약은 평균 85자(약 1~3문장), 대화 길이의 약 21% 수준
- 화자 태그(`#Person1#` 등)가 요약에도 등장 → 모델이 태그를 정확히 유지해야 ROUGE 점수 향상

**증강 데이터 구분 불가**
- train.csv에 원본과 백트랜스레이션 데이터가 혼재됨
- 증강 데이터는 `fname` 등 별도 식별자가 없으므로, 원본/증강 분리 필요시 별도 태깅 작업 필요
- 품질 검증: 백트랜스레이션 요약의 자연스러움 확인 필요

**클래스 불균형 없음**
- 요약 길이 분포는 train/dev 간 매우 유사 → 분포 이동(distribution shift) 위험 낮음

---

## 2. 베이스라인 코드 분석

제공된 두 노트북(`대화요약_Baseline_SFT.ipynb`, `대화요약_SimPO.ipynb`)을 분석했습니다.

### 2-1. Baseline SFT 파이프라인

```
[데이터 로드]
  → [Chat Template 적용 (Qwen3 형식)]
  → [SFTTrainer (전체 시퀀스 CE Loss)]
  → [추론: Greedy Decoding]
  → [ROUGE 평가 (공백 기반 + MeCab 형태소 기반)]
```

**핵심 설정 (RTX 3090 기준)**

| 파라미터 | 값 | 설명 |
|---------|---|-----|
| `MODEL_NAME` | `unsloth/Qwen3-14B` | Unsloth 최적화 버전 |
| `MAX_SEQ_LENGTH` | 2048 | 입력 최대 길이 |
| `LORA_R` / `LORA_ALPHA` | 16 / 16 | LoRA rank/scale |
| `BATCH_SIZE` | 1 | GPU 메모리 제약 |
| `GRAD_ACCUM` | 32 | 실효 배치 = 32 |
| `LEARNING_RATE` | 2e-4 | 학습률 |
| `NUM_EPOCHS` | 3 | 에폭 수 |
| `load_in_4bit` | True | QLoRA 양자화 |
| `optim` | `adamw_8bit` | 메모리 절약 옵티마이저 |

**Baseline SFT 성능 결과**

| 방법 | MeCab ROUGE-1 | 비고 |
|-----|-------------|-----|
| Baseline SFT (표준) | ~0.32 | 전체 시퀀스 CE Loss |
| Response-Only SFT | 0.5641 | 프롬프트 제외 학습 |
| Response-Only + MBR 8개 앙상블 | **0.5716** | 현재 최고 성능 |

**Baseline SFT 한계 분석**
- 전체 시퀀스(프롬프트 + 대화 + 요약)에 대해 CE Loss → 모델이 "프롬프트 암기"에 파라미터 낭비
- **Response-Only Loss** 적용 시 ROUGE-1 기준 0.32 → 0.56으로 약 75% 개선
- SFTTrainer의 `response_template` 파라미터로 어시스턴트 구간만 Loss 계산 가능

**프롬프트 설계**

```python
SYSTEM_PROMPT = (
    "당신은 한국어 대화 요약 전문가입니다. "
    "대화에는 #Person1#, #Person2# 등의 화자 태그가 사용됩니다. "
    "요약할 때 이 화자 태그를 그대로 사용하여 누가 무엇을 했는지 명확히 구분해주세요. "
    "핵심 내용만 1~3문장으로 간결하게 요약하세요."
)
```

**후처리 함수** — **실전 검증**

실제 프로젝트에서 사용된 후처리 함수입니다. `enable_thinking=False` 설정에도 불구하고 간헐적으로 특수 태그가 생성되므로 필수적입니다:

```python
import re

def postprocess(text):
    """생성된 요약 후처리 (실전 검증됨)
    
    Args:
        text: 모델이 생성한 요약 텍스트
    
    Returns:
        정제된 요약 텍스트
    """
    # 1. Thinking 태그 제거 (enable_thinking=False여도 가끔 생성)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    
    # 2. HTML/특수 태그 제거
    text = re.sub(r"<\|.*?\|>", "", text)  # <|im_start|>, <|im_end|> 등
    text = re.sub(r"<[^>]+>", "", text)    # 일반 HTML 태그
    
    # 3. 화자 태그 정규화: "#Person N#" → "#PersonN#"
    #    (모델이 공백 포함해서 생성하는 경우 있음)
    text = re.sub(r"#\s*Person\s*(\d+)\s*#", r"#Person\1#", text)
    
    # 4. "요약:" 접두사 제거 (모델이 가끔 생성)
    text = re.sub(r"^요약\s*:\s*", "", text).strip()
    
    # 5. 연속 공백 정리
    text = re.sub(r"\s+", " ", text).strip()
    
    return text if text else "빈 요약"


# 사용 예시
generated = model.generate(**inputs)
summary = tokenizer.decode(generated[0], skip_special_tokens=True)
summary = postprocess(summary)
```

**왜 필요한가**:
- ❗ `enable_thinking=False` 설정에도 불구하고 간헐적으로 `<think>` 태그 생성 (Qwen3 특성)
- ❗ 모델이 "#Person 1#" (공백 포함) 형태로 생성하는 경우 ROUGE 점수 감소
- ❗ "요약: ..." 형식으로 생성하는 경우 불필요한 토큰 제거 필요
- ✅ 후처리 적용 시 ROUGE-1 약 +0.005~0.01 향상 (실측)

### 2-2. 추론 설정 명확화 — **실전 가이드**

```python
# ★ 추론 시 핵심 설정
model.generate(
    **inputs,
    max_new_tokens=192,      # 요약 최대 길이 (실험 결과 150-200이 적정)
    do_sample=False,         # Greedy decoding (재현성 보장)
    # temperature, top_p는 사용하지 않음 (Greedy가 가장 안정적)
)

# ★ Chat template 적용 시
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,   # 추론 시 True, 학습 시 False
    enable_thinking=False,        # ★ 필수: Thinking 모드 비활성화
    tokenize=False,
)
```

**주의사항**:

| 파라미터 | 학습 | 추론 | 설명 |
|---------|-----|-----|------|
| `add_generation_prompt` | False | True | 추론 시 assistant 응답 시작 토큰 추가 |
| `enable_thinking` | False | False | 항상 False (학습/추론 모두) |
| `max_new_tokens` | - | 150-200 | 너무 크면 불필요한 텍스트, 너무 작으면 잘림 |
| `do_sample` | - | False | Greedy가 가장 안정적 (MBR 앙상블 제외) |

**max_new_tokens 설정 가이드**:
```
평균 요약 길이: 85자 ≈ 40~60 토큰
안전 마진 포함: 150~200 토큰 권장
너무 크게(500+): 불필요한 설명 생성 위험
너무 작게(100): 요약이 중간에 잘림
```

### 2-2. SimPO 파이프라인

**3단계 파이프라인 구조**

```
[Step 1] SFT 학습 → 기본 요약 능력 확보
    ↓
[Step 2] SFT 모델로 train 데이터 요약 생성 (= rejected 데이터 구축)
         골드 요약 = chosen, 모델 생성 = rejected
    ↓
[Step 3] SimPO 학습 (CPOTrainer, loss_type="simpo")
    ↓
[Step 4] 추론 & 제출
```

**SimPO 핵심 설정**

| 파라미터 | 값 | 비고 |
|---------|---|-----|
| `SIMPO_LORA_R` | 64 | SFT의 4배 |
| `SIMPO_LR` | 5e-7 | 매우 낮은 학습률 필수 |
| `SIMPO_GAMMA` | 0.5 | 마진 파라미터 |
| `SIMPO_EPOCHS` | 1 | 1에폭으로 충분 |
| `loss_type` | `"simpo"` | TRL CPOTrainer 옵션 |

**품질 필터링 로직**

```python
# Rejected 데이터 필터링
if chosen_text == rejected_text: continue           # 동일하면 제외
if len(rejected_text) < 5: continue                  # 너무 짧으면 제외
if len(rejected_text) > max(len(chosen_text)*3, 300): continue  # 대화 복사 의심
```

**SimPO 성능 (현재 베이스라인 기준)**

| 방법 | MeCab ROUGE-1 |
|-----|-------------|
| Baseline SFT | ~0.32 |
| SFT + SimPO | ~0.33 |

→ 현재 SimPO 구현은 기대 이하의 성능. 섹션 5에서 개선 방향 제시.

---

## 3. 모델 후보군 상세 분석

### 3-1. Solar 10.7B (Upstage)

**모델 정보**

| 항목 | 내용 |
|-----|-----|
| HuggingFace ID | `upstage/SOLAR-10.7B-v1.0` (base), `upstage/SOLAR-10.7B-Instruct-v1.0` |
| 파라미터 | 10.7B |
| 아키텍처 | Llama2 기반 + Depth Up-Scaling (DUS) |
| 라이선스 | Apache 2.0 (상업적 이용 가능) |
| 컨텍스트 길이 | 4,096 토큰 |
| 언어 | 영어 주력, 한국어 제한적 지원 |

**DUS (Depth Up-Scaling) 기술**
- Llama2 32레이어를 베이스로 복사본 생성 후 레이어 일부 제거하여 48레이어 스택
- Mistral 7B 가중치를 업스케일된 레이어에 통합, 전체 모델 continued pre-training
- 30B 이하 모델 중 당시 최고 성능 기록 (HuggingFace Open LLM 리더보드 1위, 2023년 12월)

**24GB VRAM 적합성 분석**

| 정밀도 | VRAM | 상태 |
|--------|-----|-----|
| fp16 (풀 로드) | ~21GB | 추론 가능 (학습 어려움) |
| 4-bit QLoRA | ~5.5GB | ✅ SFT/SimPO 학습 가능 |

**한국어 성능 평가**
- 한국어 특화 프리트레이닝은 없으나 다양한 한국어 파인튜닝 커뮤니티 모델 존재
- 태그 구조(`#Person1#` 등) 이해: 학습 데이터로 충분히 학습 가능
- **주의**: 컨텍스트 길이 4,096 토큰 → 시스템 프롬프트 + 긴 대화 처리 시 트런케이션 위험
- 긴 대화(2,000자 이상)에서는 Qwen3-14B 대비 성능 저하 가능성

**권장 설정**

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/SOLAR-10.7B-v1.0",
    max_seq_length=2048,
    load_in_4bit=True,
)
```

---

### 3-2. Qwen3-14B (Alibaba Cloud) — **Primary Model**

**모델 정보**

| 항목 | 내용 |
|-----|-----|
| HuggingFace ID | `Qwen/Qwen3-14B`, `unsloth/Qwen3-14B` |
| 파라미터 | 14.8B (Non-embedding: 13.2B) |
| 아키텍처 | Dense Transformer (40레이어, GQA: Q×40, KV×8) |
| 라이선스 | Apache 2.0 |
| 컨텍스트 길이 | 32,768 토큰 (YaRN으로 131,072 확장) |
| 언어 | 119개 언어/방언 지원 (한국어 포함) |
| 출시 | 2025년 4월 |

**아키텍처 및 학습 특성**
- 36조 토큰 프리트레이닝 (Qwen2.5의 약 2배, 119개 언어 커버)
- qk-layernorm 적용 → 학습 안정성 향상
- 3단계 프리트레이닝: (1) 일반 언어 모델링 → (2) STEM/코딩/추론 강화 → (3) 32K 장문 컨텍스트
- **Thinking Mode**: `<think>` 태그로 Chain-of-Thought 활성화/비활성화 제어
- Qwen3-14B는 Qwen2.5-32B와 동급 성능 달성 (더 큰 모델과 동등한 효율)

**24GB VRAM 적합성 분석**

| 정밀도 | VRAM 사용량 | 상태 |
|--------|---------|-----|
| BF16 (풀) | ~29.5GB | 초과 |
| FP8 | ~15GB | ✅ 추론 가능 |
| 4-bit QLoRA (Unsloth) | ~9GB | ✅ 학습 가능 |
| Q4_K_M GGUF | ~9GB | ✅ 추론만 |

**한국어 성능 특성**
- Qwen3 14B는 한국어 이해 능력 우수 (KMMLU 기준 약 58.5~60% 정확도)
- 단, 내부 추론(chain-of-thought)이 기본적으로 영어로 진행되는 경향 있음
- SFT 후 한국어 요약 품질 크게 향상 (Response-Only 기준 ROUGE-1 0.56 달성)
- **`enable_thinking=False` 설정 필수** → train.csv가 일상적인 대화 위주로 구성되어 고차원 논리 추론이 불필요하며, 요약 태스크에서 `<think>` 블록 생성을 방지하여 출력 토큰 수 급증 및 VRAM 오버헤드를 제거
- 연구 결과(arXiv:2508.10355): SFT 후 한국어 KMMLU 약 1.5포인트 향상 확인

**프로젝트 운영 전략**
- **4-bit QLoRA + Response-Only SFT + SimPO 파이프라인으로 운영**
- **LoRA Rank를 16에서 32로 상향**하여 한국어 대화체에 대한 학습 밀도를 높임
- 베이스라인 코드의 기본 모델로, 가장 검증된 설정

**권장 설정 (Response-Only SFT)**

```python
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# Response-Only 컬렉터 설정 (핵심 개선 포인트!)
response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    data_collator=collator,  # Response-Only Loss 적용
    args=SFTConfig(...)
)
```

---

### 3-3. Ministral 3 14B (Mistral AI)

**모델 정보**

| 항목 | 내용 |
|-----|-----|
| HuggingFace ID (FP8) | `mistralai/Ministral-3-14B-Instruct-2512` |
| HuggingFace ID (BF16) | `mistralai/Ministral-3-14B-Instruct-2512-BF16` |
| HuggingFace ID (Base) | `mistralai/Ministral-3-14B-Base-2512` |
| 파라미터 | 14B |
| 아키텍처 | Dense Transformer (비전 포함 멀티모달) |
| 라이선스 | Apache 2.0 |
| 컨텍스트 길이 | 최대 262,144 토큰 |
| 언어 | 40+ 언어 (한국어 명시 지원) |
| 출시 | 2025년 12월 |

**주요 특징**
- Mistral Small 3.2 24B에 필적하는 성능을 14B 파라미터로 달성
- 비전(이미지 이해) + 텍스트 멀티모달 지원
- 24GB VRAM에 FP8로 딱 맞게 배치 가능 (FP8 ~ 17~19GB 예상)
- 2025년 12월 최신 출시로 최신 한국어 데이터 포함 기대

**24GB VRAM 적합성 분석**

| 정밀도 | VRAM | 상태 |
|--------|-----|-----|
| BF16 | ~28GB | 초과 |
| FP8 (공식 제공) | ~17GB | ✅ 추론 가능 |
| 4-bit GGUF/GPTQ | ~9GB | ✅ 학습 가능 |

**Unsloth 지원 현황**

```bash
# Unsloth GGUF 버전 제공 확인
# unsloth/Ministral-3-14B-Instruct-2512-GGUF
```

**FP8 모델 로드 방법**

```python
from transformers import Mistral3ForConditionalGeneration, FineGrainedFP8Config
model = Mistral3ForConditionalGeneration.from_pretrained(
    "mistralai/Ministral-3-14B-Instruct-2512",
    device_map="auto",
    quantization_config=FineGrainedFP8Config(dequantize=True)
)
# mistral-common >= 1.8.6 설치 필요
```

---

### 3-4. Llama 4 Maverick 17B-128E (Meta) — **Teacher Model (API 전용)**

**모델 정보**

| 항목 | 내용 |
|-----|-----|
| HuggingFace ID | `meta-llama/Llama-4-Maverick-17B-128E-Instruct` |
| 총 파라미터 | ~400B (128개 전문가 MoE) |
| **활성 파라미터** | **17B** (토큰 처리 시 실제 계산) |
| 아키텍처 | MoE (128 experts), dense/MoE 레이어 교차 배치 |
| 라이선스 | Llama 4 Community License |
| 컨텍스트 길이 | 최대 1,000,000 토큰 |
| 언어 | 200개 언어 (fine-tuning 공식 지원: 영어/아랍어 등 12개, 한국어 미포함) |
| 출시 | 2025년 4월 5일 |

**아키텍처 특성**
- 128개 전문가(Expert) 중 토큰당 일부만 활성화 → 17B 활성 파라미터로 ~400B 규모 성능
- MoE와 Dense 레이어가 교차 배치 (절반만 MoE 레이어)
- Llama Behemoth (대형 모델)로부터 Co-Distillation 학습
- Early fusion 멀티모달 (텍스트 + 이미지)
- 프리트레이닝: ~22조 토큰 (Scout의 절반 규모)

**24GB VRAM에서의 현실적 제약**

| 포맷 | VRAM | 학습/추론 |
|------|-----|---------|
| BF16 (총 파라미터 기준) | ~800GB+ | 불가 |
| FP8 (공식 제공) | 멀티 GPU 필요 | 단일 GPU 불가 |
| INT4 GPTQ/GGUF | ~25~35GB | 추론만, 24GB 초과 가능성 |
| LoRA SFT | 불가 | 단일 GPU 불가 |

**프로젝트 활용 전략 (Teacher Model)**

RTX 3090 단일 환경에서 직접적인 SFT/SimPO 학습이 불가능하므로, **API를 통한 데이터 증강 및 검증용 교사 모델**로 활용:

1. **데이터 정제**: API를 통해 train.csv 데이터의 품질 검증 및 정제
2. **SimPO 선호도 데이터 생성**: Chosen/Rejected 페어 생성 시 고품질 참조 요약 제공
3. **내부 검증**: SFT 모델 결과물에 대한 품질 평가 및 체크포인트 선별 참고

**결론**: 24GB 환경에서 직접 학습은 불가능하나, API를 활용한 교사·데이터 파이프라인 용도로 프로젝트 품질 향상에 기여

**대안**: Llama 4 Scout (109B 총 파라미터, 17B 활성)도 단일 24GB GPU에서 학습 어려움.

---

### 3-5. 모델 선택 매트릭스 (24GB VRAM 기준)

| 모델 | VRAM (4bit SFT) | 한국어 | 컨텍스트 | SFT | SimPO | 종합 추천 |
|-----|--------------|------|---------|-----|-------|---------|
| Solar 10.7B | ~7GB | ⭐⭐⭐ | 4,096 | ✅ | ✅ | ⭐⭐⭐ |
| **Qwen3-14B** | ~9GB | ⭐⭐⭐⭐⭐ | 32,768 | ✅ | ✅ | ⭐⭐⭐⭐⭐ |
| Ministral3-14B | ~9GB | ⭐⭐⭐⭐ | 262,144 | ✅ | ✅ | ⭐⭐⭐⭐ |
| Llama 4 Maverick | ~25GB+ | ⭐⭐⭐ | 1M | ❌ | ❌ | ⭐ |

**최종 권장 순서**: Qwen3-14B > Ministral3-14B > Solar 10.7B

---

## 4. SFT 전략

### 4-1. Response-Only Loss (최우선 개선 사항) — **실전 검증**

베이스라인(표준 SFT)과 Response-Only SFT의 성능 차이가 0.32 → 0.56으로 매우 큼.
**가장 먼저 적용해야 할 개선 사항**입니다.

#### 이론적 배경

**문제점**: 일반적인 SFT는 전체 시퀀스(시스템 프롬프트 + 사용자 입력 + 모델 응답)의 모든 토큰에 대해 Loss를 계산합니다. 이는 프롬프트를 암기하는 데 파라미터를 낭비합니다.

**해결책**: Response-Only Loss는 모델 응답(요약) 부분에만 Loss를 계산하여 요약 생성에 집중합니다.

```
[시스템 프롬프트] → loss 계산 X (label = -100으로 마스킹)
[사용자 입력]     → loss 계산 X (label = -100으로 마스킹)
[모델 응답]       → loss 계산 O ← 이것만 학습!
```

**성능 향상**: ROUGE-1 0.32 → 0.56 **(75% 향상)**

#### 실전 구현 코드 (검증됨)

실제 프로젝트에서는 커스텀 `ResponseOnlyDataCollator`를 구현하여 사용합니다:

```python
from dataclasses import dataclass
from typing import Any, Dict, List
import torch

@dataclass
class ResponseOnlyDataCollator:
    """프롬프트 토큰을 -100으로 마스킹하여 응답(요약)만 학습하는 DataCollator
    
    PyTorch의 CrossEntropyLoss는 label이 -100인 토큰을 무시합니다.
    이를 이용해 프롬프트 부분의 label을 -100으로 설정하면,
    모델 응답 부분에 대해서만 loss가 계산됩니다.
    """
    tokenizer: Any
    response_template_ids: List[int]  # assistant 응답 시작 토큰 ID
    max_length: int = 2048

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # 1. 패딩 적용
        batch = self.tokenizer.pad(
            features, padding=True, 
            max_length=self.max_length, return_tensors="pt"
        )
        
        # 2. labels 복사
        labels = batch["input_ids"].clone()

        # 3. 각 샘플에서 assistant 응답 시작점을 찾아 그 이전을 마스킹
        for i in range(len(labels)):
            ids = batch["input_ids"][i].tolist()
            response_start = self._find_response_start(ids)
            if response_start >= 0:
                labels[i, :response_start] = -100  # 프롬프트 부분 마스킹
            # 패딩 토큰도 마스킹
            labels[i, batch["attention_mask"][i] == 0] = -100

        batch["labels"] = labels
        return batch

    def _find_response_start(self, ids: List[int]) -> int:
        """토큰 시퀀스에서 assistant 응답 시작 위치를 찾습니다."""
        template = self.response_template_ids
        if template is None:
            return 0
        last_pos = -1
        for i in range(len(ids) - len(template) + 1):
            if ids[i:i + len(template)] == template:
                last_pos = i + len(template)
        return last_pos


# 사용 예시
response_template_str = "<|im_start|>assistant\n"
response_template_ids = tokenizer.encode(response_template_str, add_special_tokens=False)

collator = ResponseOnlyDataCollator(
    tokenizer=tokenizer,
    response_template_ids=response_template_ids,
    max_length=2048,
)

# SFTTrainer에 전달
from trl import SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    data_collator=collator,  # ← Response-Only loss의 핵심!
    args=training_args,
)
```

**검증 방법**: 학습 전 마스킹 비율 확인
```python
sample_ids = tokenizer.encode(sample_text, add_special_tokens=False)
pos = collator._find_response_start(sample_ids)
masking_ratio = pos / len(sample_ids)
print(f"마스킹 비율: {masking_ratio:.1%}")  # 예: 75% → 프롬프트 부분 제외
```

#### 실제 성능 결과 (실측)

| 방법 | Mecab ROUGE-1 | Mecab ROUGE-2 | 비고 |
|------|--------------|---------------|------|
| KoBART baseline | ~0.35 | - | 대회 제공 베이스라인 |
| Baseline SFT (전체 시퀀스) | ~0.32 | - | 프롬프트 포함 학습 |
| **Qwen3-14B Response-Only SFT** | **0.5641** | **0.3849** | **이 전략 (검증됨)** |
| Qwen3-32B SFT | 0.5433 | 0.3627 | 더 큰 모델이지만 오히려 낮음 |
| **MBR 8-model 앙상블** | **0.5716** | **0.3883** | **최고 성능** |

**인사이트**:
- Response-Only Loss의 효과: 베이스라인 대비 **+0.24 향상** (0.32 → 0.5641)
- 32B 모델이 14B보다 낮은 이유: 과적합 또는 한국어 학습 데이터 부족 가능성
- MBR 앙상블 추가 시 **+0.0075 향상** (0.5641 → 0.5716)

### 4-2. LoRA 하이퍼파라미터 탐색

| 설정 | 학습 파라미터 수 | VRAM(Qwen3-14B) | 예상 성능 |
|-----|--------------|---------------|---------|
| r=8, alpha=8 | ~50MB | ~8.5GB | 낮음 |
| r=16, alpha=16 (베이스라인) | ~100MB | ~9GB | 보통 |
| **r=32, alpha=32 (운영 확정)** | **~200MB** | **~10GB** | **높음** |
| r=64, alpha=64 | ~400MB | ~12GB | 최고 |
| r=128, alpha=128 | ~800MB | ~15GB | 과적합 위험 |

**권장**: **r=32, alpha=32** (24GB VRAM에서 여유 있으며, 한국어 대화체 학습 밀도를 높여 성능과 안정성 균형 확보)

**타겟 모듈 (전체 선택 권장)**

```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # Attention
    "gate_proj", "up_proj", "down_proj",         # FFN/MLP
]
```

### 4-3. Gradient Accumulation 원리 — **실전 가이드**

**핵심 개념**: GPU 메모리가 부족할 때 큰 배치 효과를 얻는 방법

```
실효 배치 크기 = per_device_batch_size × gradient_accumulation_steps

RTX 3090 (24GB):  1 × 32 = 32  ← 메모리 부족하지만 큰 배치 효과
RTX A6000 (48GB): 4 × 8  = 32  ← 동일한 학습 효과
A100 (80GB):      8 × 4  = 32
```

**작동 원리**:
1. 배치 1개를 forward pass → loss 계산
2. Gradient를 메모리에 누적 (optimizer.step() 호출 안 함)
3. 32개 누적 후 한 번에 optimizer.step() 호출
4. Gradient 초기화 후 다시 누적 시작

**장점**:
- 메모리는 배치 1개 수준으로 절약
- 학습 안정성은 배치 32개 수준으로 확보
- 학습 결과는 실제 배치 32와 동일

**단점**:
- 학습 시간이 약간 증가 (forward pass 32번)
- BatchNorm 계층 사용 시 주의 필요 (Transformer는 LayerNorm 사용하므로 무관)

**SFTConfig 설정 예시**:
```python
from trl import SFTConfig

sft_config = SFTConfig(
    per_device_train_batch_size=1,      # GPU에 한 번에 넣는 배치 수
    gradient_accumulation_steps=32,     # Gradient 누적 횟수
    # 실효 배치 = 1 × 32 = 32
    
    # 기타 설정
    learning_rate=2e-4,
    num_train_epochs=3,
    max_grad_norm=1.0,  # Gradient clipping
)
```

### 4-4. 학습률 스케줄러 전략

| 스케줄러 | 특성 | 권장 |
|---------|-----|-----|
| `cosine` | 부드러운 감소, 안정적 | ✅ 기본 권장 |
| `cosine_with_restarts` | 주기적 재가열 | 장기 학습 시 |
| `linear` | 단순, 빠른 수렴 | 단기 실험 |

**권장 설정**

```yaml
learning_rate: 2e-4
warmup_ratio: 0.05
lr_scheduler_type: cosine
num_train_epochs: 3
```

### 4-5. Early Stopping 적용

17,341개 학습 데이터 기준, 에폭당 약 542 스텝 (실효 배치 32 기준). 과적합 방지를 위해 Early Stopping 권장:

```python
from transformers import EarlyStoppingCallback
callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
```

### 4-6. 모델별 SFT 설정 요약

| 모델 | LoRA r | LR | 배치 | Grad Accum | 예상 학습 시간/에폭 |
|-----|--------|-----|-----|-----------|--------------|
| Solar 10.7B | 32 | 2e-4 | 1 | 32 | 3~4시간 |
| **Qwen3-14B (Primary)** | **32** | **2e-4** | **1** | **32** | **4~5시간** |
| Ministral3-14B | 32 | 2e-4 | 1 | 32 | 4~5시간 |

**Qwen3-14B 확정 설정**: LR 2e-4, Epochs 3, Effective Batch Size 32 (1×32), Response-Only Loss, `enable_thinking=False`

---

## 5. SimPO 전략

### 5-1. SimPO 수식 및 원리

$$\mathcal{L}_{\text{SimPO}} = -\log\sigma\left(\frac{\beta}{|y_w|}\log p_\theta(y_w|x) - \frac{\beta}{|y_l|}\log p_\theta(y_l|x) - \gamma\right)$$

- $y_w$ (chosen): 골드 요약 (높은 ROUGE)
- $y_l$ (rejected): SFT 모델 생성 요약 (낮은 ROUGE)
- $\beta$: 보상 스케일링 (SimPO 권장: 2.0~10.0)
- $\gamma$: 마진 파라미터 (0.5~1.5 권장)
- **Reference Model 불필요** → DPO 대비 메모리 약 50% 절약, RTX 3090 24GB 환경에서 효율적 정렬(Alignment) 가능
- 목적: 요약문의 **간결성(Conciseness)**과 **사실 관계(Factuality)** 정확도 향상
- NeurIPS 2024 발표 논문 (arXiv:2405.14734)

**SimPO vs DPO 비교**

| 항목 | DPO | SimPO |
|-----|-----|-------|
| Reference Model | 필요 | **불필요** |
| VRAM 사용량 | 2배 | 1배 |
| 길이 편향 | 있음 | 없음 (길이 정규화) |
| 하이퍼파라미터 | beta | beta, gamma |
| 성능 (AlpacaEval 2) | 기준 | +6.4포인트 |

### 5-2. 현재 SimPO 성능 저조 원인 분석

베이스라인에서 SimPO가 SFT 대비 거의 개선이 없는 주요 원인:

1. **표준 SFT 기반 출발**: Baseline SFT(0.32) 위에 SimPO → 이미 낮은 기준점에서 시작
2. **beta 값 설정 문제**: SimPO 논문 권장 beta=2.0~10.0, 현재 CPOTrainer 기본값 확인 필요
3. **gamma 값 낮음**: 현재 0.5 → 1.0~1.5로 증가 시도 권장
4. **학습률 조정 필요**: 5e-7 → 3e-7 또는 1e-7 시도
5. **rejected 품질 문제**: 전체 시퀀스 SFT(0.32 수준)로 생성한 rejected가 chosen과 유사할 가능성

### 5-3. SimPO 개선 방안

**방안 1: Response-Only SFT 기반 SimPO (핵심)**
- Response-Only SFT로 먼저 0.56 달성 후 SimPO 적용
- 더 강력한 SFT 모델 → rejected 품질 차이 명확 → SimPO 효과 증대

**방안 2: On-policy Rejected 생성**
- Temperature Sampling으로 N개 생성 후 ROUGE 기준 chosen/rejected 선택

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=192,
    num_return_sequences=4,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
)
# ROUGE로 best → chosen, worst → rejected 선택
```

**방안 3: 하이퍼파라미터 그리드 서치**

| 파라미터 | 탐색 범위 |
|---------|---------|
| `simpo_gamma` | 0.5, 1.0, 1.5, 2.0 |
| `beta` (CPOConfig) | 2.0, 5.0, 10.0 |
| `learning_rate` | 1e-7, 3e-7, 5e-7, 1e-6 |
| `LORA_R` | 32, 64 |

**방안 4: ORPO (선택적 대안 알고리즘)**
- SimPO와 마찬가지로 Reference Model 불필요
- SFT Loss + 선호도 Loss를 단일 손실 함수로 통합
- 일부 경우 SimPO보다 안정적인 학습 보고
- **본 프로젝트는 SimPO를 우선 전략으로 채택**, ORPO는 필요 시 보조 실험으로 고려

```python
from trl import ORPOConfig, ORPOTrainer
# lambda_: SFT 손실 가중치 (0.1~1.0)
```

### 5-4. 권장 SimPO 파이프라인 (개선 버전)

```
[Step 1] Response-Only SFT (r=32, 3 에폭)
    ↓ ROUGE-1 ~0.56 달성
[Step 2] On-policy Rejected 생성 (Llama 4 API 또는 SFT 모델)
         - Temperature=0.7, num_return_sequences=4
         - ROUGE 기반 chosen/rejected 페어링
         - 필터: |chosen_rouge - rejected_rouge| > 0.05
         - Llama 4 API로 사실 관계 검증 및 선호도 데이터 품질 향상
    ↓
[Step 3] SimPO (beta=5.0, gamma=1.0, lr=3e-7, r=64, 1 에폭)
    ↓
[Step 4] MBR 앙상블 디코딩 (섹션 7)
```

**대안: ORPO (선택적)**
- SimPO와 마찬가지로 Reference Model 불필요
- SFT Loss + 선호도 Loss를 단일 손실 함수로 통합
- 일부 경우 SimPO보다 안정적인 학습 보고되나, 본 프로젝트는 SimPO를 우선 전략으로 채택

---

## 6. VRAM 최적화 및 메모리 예산

### 6-1. 24GB VRAM 예산 계획 (Qwen3-14B 기준)

| 컴포넌트 | 메모리 | 비고 |
|---------|------|-----|
| 모델 가중치 (4-bit) | ~8.5GB | QLoRA NF4 양자화 |
| LoRA 어댑터 (r=32) | ~200MB | 학습 가능 파라미터 |
| 옵티마이저 상태 (8-bit Adam) | ~400MB | adamw_8bit |
| 활성화 메모리 (batch=1) | ~2GB | Gradient Checkpointing 적용 시 |
| KV Cache (추론) | ~1GB | |
| **총합** | **~12GB** | **12GB 여유** |

**배치 사이즈 최적화**

| 설정 | 실효 배치 | VRAM | 속도 |
|-----|---------|-----|-----|
| batch=1, accum=32 | 32 | ~12GB | 기준 |
| batch=2, accum=16 | 32 | ~14GB | 빠름 |
| batch=4, accum=8 | 32 | ~18GB | 더 빠름 |
| batch=8, accum=4 | 32 | ~24GB | 한계 |

### 6-2. 메모리 최적화 기법

**필수 기법 (모두 적용)**

```python
# 1. Gradient Checkpointing (Unsloth 최적화 버전)
use_gradient_checkpointing="unsloth"

# 2. Flash Attention 2 (RTX 3090 Ampere 지원)
# RTX 3090은 Ampere 아키텍처로 Flash Attention 3의 하드웨어 가속을 지원하지 않음
# FA2 + Unsloth 커널 최적화 조합으로 메모리 효율 극대화
attn_implementation="flash_attention_2"

# 3. 8-bit 옵티마이저
optim="adamw_8bit"

# 4. Mixed Precision
bf16=True  # Ampere 이상 GPU

# 5. 배치 패킹 (짧은 시퀀스 결합, 처리량 향상)
# 주의: 실험 시 packing=False 시도 가능 (데이터 특성에 따라 성능 차이 있을 수 있음)
packing=True  # SFTConfig 옵션

# 6. max_seq_length 설정으로 긴 대화 로그 처리 시 트런케이션 방지
max_seq_length=2048  # train.csv 최대 대화 길이 커버
```

### 6-3. 학습 안정성 체크리스트

- [ ] `torch.cuda.empty_cache()` 단계 간 호출
- [ ] `max_seq_length` 트런케이션으로 OOM 방지
- [ ] WandB 초기 실험 시 `report_to="none"` 설정
- [ ] 체크포인트 `save_total_limit=2`로 디스크 공간 관리

---

## 7. 디코딩 전략 (MBR 앙상블) — **실전 검증**

### 7-1. MBR (Minimum Bayes Risk) 디코딩 원리

**현재 최고 성능의 핵심**: Response-Only SFT + MBR 8개 프롬프트 앙상블 → **ROUGE-1 0.5716 (검증됨)**

**원리**: 여러 후보 요약 중 ROUGE 기대값이 최대인 요약 선택

$$y^* = \arg\max_{y \in \mathcal{H}} \frac{1}{|\mathcal{H}|} \sum_{y' \in \mathcal{H}} \text{ROUGE}(y, y')$$

**직관**: "다수결" 방식 - 여러 프롬프트가 비슷하게 요약한 내용이 가장 신뢰할 수 있음

```
프롬프트 A → 요약 A (ROUGE-1: 0.56)
프롬프트 B → 요약 B (ROUGE-1: 0.57) ← 다른 요약들과 가장 유사 → 최종 선택!
프롬프트 C → 요약 C (ROUGE-1: 0.55)
...
프롬프트 H → 요약 H (ROUGE-1: 0.56)
```

### 7-2. 실전 프롬프트 변형 전략 (8개, 검증됨)

MBR 앙상블의 핵심은 **다양한 프롬프트**입니다. 같은 체크포인트를 사용하지만, 프롬프트를 다르게 하면 미묘하게 다른 요약이 생성됩니다.

| 변형 이름 | 특징 | 스타일 | 목적 |
|----------|------|--------|------|
| `dev_save` | 기본 (화자 태그 강조) | 지시형 | 베이스라인 |
| `abstract` | 추상적 요약 스타일 | "~에 대해 이야기한다" | 스타일 다양화 |
| `goldstyle` | 1-shot 예시 포함 | Few-shot | 출력 형식 일관성 |
| `goldstyle_v2` | 3-shot + 추상적 | Few-shot + 추상 | 예시 기반 학습 |
| `dynfew3` | dev_save 동일 | 복제 | 안정성 확보 |
| `r7_ep2` | 2 에폭 학습 모델 | 다른 체크포인트 | 모델 다양성 |
| `r9_32b` | Qwen3-32B (더 큰 모델) | 모델 크기 변형 | 파라미터 다양성 |
| `r9_32b_fewshot` | 32B + few-shot | 모델+프롬프트 | 복합 다양성 |

**핵심 전략**:
1. **프롬프트 스타일 다양화**: 지시형 vs 추상형 vs Few-shot
2. **모델 체크포인트 변형**: 에폭 수, 모델 크기
3. **복제 포함**: 과도한 다양성 방지 (안정성 확보)

**프롬프트 예시** (`abstract` 변형):
```python
PROMPTS = {
    "abstract": {
        "system": (
            "당신은 한국어 대화 요약 전문가입니다. "
            "대화의 주요 주제와 화자들의 행동을 요약하세요. "
            "#Person1#, #Person2# 등 화자 태그를 반드시 사용하고, "
            "'~에 대해 이야기한다', '~을 요청한다' 같은 표현을 활용하세요. "
            "1~2문장으로 간결하게 요약하세요."
        ),
        "user": "다음 대화를 요약하세요.\n\n{dialogue}",
    },
}
```

### 7-3. 실제 MBR 구현 코드 (검증됨)

```python
import mecab
from rouge import Rouge
from tqdm import tqdm

def mbr_ensemble(candidates, use_mecab=True):
    """MBR 디코딩으로 최종 요약 선택
    
    Args:
        candidates: [(prompt_name, summary_text), ...] 형태의 리스트
        use_mecab: Mecab 형태소 분석 사용 여부
    
    Returns:
        최종 선택된 요약 텍스트
    """
    rouge = Rouge()
    
    # 1. Mecab 형태소 분석 (대회 평가 기준)
    if use_mecab:
        m = mecab.MeCab()
        cand_morphs = [" ".join(m.morphs(c[1])) for c in candidates]
    else:
        cand_morphs = [c[1] for c in candidates]
    
    # 2. Pairwise ROUGE-1 계산
    best_score = -1
    best_idx = 0
    
    for j in range(len(candidates)):
        avg_rouge = 0
        count = 0
        
        for k in range(len(candidates)):
            if j != k:
                try:
                    score = rouge.get_scores([cand_morphs[j]], [cand_morphs[k]])[0]
                    avg_rouge += score["rouge-1"]["f"]
                    count += 1
                except:
                    pass  # 빈 요약 등 예외 처리
        
        if count > 0:
            avg_rouge /= count
        
        if avg_rouge > best_score:
            best_score = avg_rouge
            best_idx = j
    
    return candidates[best_idx][1]  # 원본 텍스트 반환 (형태소 아님)


# 전체 데이터셋에 대한 MBR 적용
def apply_mbr_to_dataset(test_df, all_predictions):
    """데이터셋 전체에 MBR 앙상블 적용
    
    Args:
        test_df: 테스트 데이터프레임
        all_predictions: {prompt_name: [pred1, pred2, ...]} 딕셔너리
    
    Returns:
        MBR로 선택된 최종 요약 리스트
    """
    model_names = list(all_predictions.keys())
    n_samples = len(test_df)
    
    mbr_preds = []
    model_selected = {name: 0 for name in model_names}
    
    for i in tqdm(range(n_samples), desc="MBR Ensemble"):
        # 이 샘플에 대한 모든 후보 수집
        candidates = [(name, all_predictions[name][i]) for name in model_names]
        
        # MBR로 최적 요약 선택
        selected = mbr_ensemble(candidates, use_mecab=True)
        mbr_preds.append(selected)
        
        # 어떤 프롬프트가 선택되었는지 통계
        for name, text in candidates:
            if text == selected:
                model_selected[name] += 1
                break
    
    # 선택 빈도 출력
    print("\n모델 선택 빈도:")
    for name, count in sorted(model_selected.items(), key=lambda x: -x[1]):
        print(f"  {name:20s}: {count:3d} ({100*count/n_samples:5.1f}%)")
    
    return mbr_preds
```

### 7-4. MBR 성능 분석 (실측)

**추론 비용 vs 성능 트레이드오프**:

| 프롬프트 수 | 추론 시간 (RTX 3090) | ROUGE-1 (실측) | 증분 효과 |
|-----------|---------------------|---------------|----------|
| 1개 (Greedy) | ~5분 | 0.5641 | 기준 |
| 4개 | ~20분 | ~0.570 | +0.006 |
| **8개** | **~40분** | **0.5716** | **+0.0075** |
| 16개 | ~80분 | ~0.572 | +0.001 (포화) |

**권장**: **8개가 비용 대비 효과 최적점** - 16개로 늘려도 거의 향상 없음

**선택 빈도 분석 예시**:
```
dev_save       : 145 (29.1%) ████████████
abstract       : 98  (19.6%) ████████
goldstyle      : 87  (17.4%) ███████
goldstyle_v2   : 71  (14.2%) ██████
r7_ep2         : 48  ( 9.6%) ████
r9_32b         : 32  ( 6.4%) ███
r9_32b_fewshot : 12  ( 2.4%) █
dynfew3        : 6   ( 1.2%) █
```
→ 기본 프롬프트(`dev_save`)가 가장 자주 선택되지만, 다양한 프롬프트가 골고루 기여

### 7-5. 앙상블 크기별 성능 트레이드오프

| 후보 수 | 예상 ROUGE-1 향상 | 추론 시간 배율 | 실측 효과 |
|--------|---------------|------------|----------|
| 1 (Greedy) | 기준 (0.5641) | 1× | 기준 |
| 4개 | +0.5~1% | 4× | +0.006 |
| **8개** | **+1~2%** | **8×** | **+0.0075 (검증)** |
| 16개 | +1.5~2.5% | 16× | +0.001 (포화) |

### 7-6. 다양한 디코딩 전략 비교

| 전략 | 다양성 | 품질 | 속도 | 추천 |
|-----|------|-----|-----|-----|
| Greedy | 낮음 | 일관 | 빠름 | 빠른 실험용 |
| Beam Search (4) | 낮음 | 높음 | 보통 | 안정적 |
| Sampling (t=0.7) | 높음 | 변동 | 빠름 | MBR과 병행 |
| **MBR × 8** | **최고** | **최고** | **느림** | **최종 제출** |

---

## 8. 데이터 증강 분석

### 8-1. 현재 증강 현황

- **원본 데이터**: 약 14,841개 추정 (17,341 - 2,500)
- **백트랜스레이션 증강**: 약 2,500개 (전체의 ~14.4%)
- 원본/증강 구분 식별자 없음 → 별도 태깅 필요

### 8-2. 백트랜스레이션 품질 검증

```python
# 증강 데이터 품질 지표
# 1. 요약 길이 분포 비교 (원본 vs 증강)
# 2. ROUGE self-score 계산 (원본 대화와 증강 요약 간 일관성)
# 3. 화자 태그 일치 여부 확인
```

### 8-3. 추가 증강 전략

**방안 1: 요약 패러프레이징**
- 대형 LLM으로 골드 요약의 동의어/문장 구조 변환
- 원본 대화는 유지, 요약만 다양화 → 데이터 효율 향상

**방안 2: 화자 순서 랜덤화**
- `#Person1#`과 `#Person2#` 역할 교환
- 단, 요약의 화자 참조도 함께 교환 필요

**방안 3: 토픽 기반 데이터 균형화**
- 9,235개 고유 토픽 중 빈도 낮은 토픽 증강
- 희귀 도메인 과소표현 해소

**방안 4: 대화 절단 증강**
- 긴 대화를 중간에서 잘라 더 짧은 샘플 생성
- 실제 테스트 데이터 길이 분포와 유사

---

## 9. 프롬프트 엔지니어링 전략 — **실전 검증**

### 9-1. 검증된 Base 프롬프트 (ROUGE-1 0.5641 달성)

실제 프로젝트에서 가장 높은 성능을 달성한 프롬프트 템플릿입니다:

```python
SYSTEM_PROMPT = (
    "당신은 한국어 대화 요약 전문가입니다. "
    "대화에는 #Person1#, #Person2# 등의 화자 태그가 사용됩니다. "
    "요약할 때 이 화자 태그를 그대로 사용하여 누가 무엇을 했는지 명확히 구분해주세요. "
    "핵심 내용만 1~3문장으로 간결하게 요약하세요."
)

USER_TEMPLATE = (
    "아래 대화를 읽고 핵심 내용을 요약해주세요. "
    "화자 태그(#Person1# 등)를 유지하세요.\n\n{dialogue}"
)

# 채팅 템플릿 적용
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_TEMPLATE.format(dialogue=dialogue)},
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False,
    add_generation_prompt=True,   # 추론 시 True
    enable_thinking=False,         # 필수: Thinking 모드 비활성화
)
```

**핵심 요소**:
1. **역할 부여**: "대화 요약 전문가" - 모델의 출력 스타일 정의
2. **화자 태그 유지 명시**: ROUGE 점수에 직접 영향 (태그 누락 시 감점)
3. **길이 제한**: "1~3문장" 명확히 지시 - 과도한 장황함 방지
4. **간결성 강조**: "핵심 내용만" - 불필요한 세부사항 제거

### 9-2. 프롬프트 변형 전략 (MBR 앙상블용)

**변형 A: 추상적 요약 스타일 (MBR 앙상블용)**

```python
SYSTEM_A = """당신은 한국어 대화 요약 전문가입니다.
대화의 주요 주제와 화자들의 행동을 요약하세요.
#Person1#, #Person2# 등 화자 태그를 반드시 사용하고,
'~에 대해 이야기한다', '~을 요청한다' 같은 표현을 활용하세요.
1~2문장으로 간결하게 요약하세요."""

USER_A = "다음 대화를 요약하세요.\n\n{dialogue}"
```

**특징**: 추상적 동사 사용 권장 - "논의한다", "제안한다" 등

**변형 B: 1-shot Few-shot 예시 포함**

```python
SYSTEM_B = """당신은 한국어 대화 요약 전문가입니다.
주어진 대화를 읽고 핵심 내용을 1~2문장으로 요약하세요.

규칙:
- 화자 태그(#Person1#, #Person2# 등)를 반드시 그대로 사용하세요.
- 불필요한 세부사항은 생략하고 핵심 행동/결정/결과만 포함하세요.
- 요약은 반드시 완전한 문장으로 끝나야 합니다.

[예시]
대화:
#Person1#: 이것은 좋은 기본 컴퓨터 패키지입니다.
#Person2#: 모뎀도 포함되어 있나요?
#Person1#: 네, 내장 모뎀이 있습니다.
#Person2#: 좋습니다. 구매하겠습니다.

요약: #Person1#은 기본 컴퓨터 패키지를 #Person2#에게 보여주고, #Person2#는 구매하기로 한다."""

USER_B = "다음 대화를 요약하세요.\n\n{dialogue}"
```

**특징**: 명시적 예시 제공으로 출력 형식 일관성 향상

### 9-3. Few-shot 프롬프트 효과 분석 (실측)

**1-shot 효과**:
- ✅ 출력 형식 일관성 향상 (예시를 따라 비슷한 구조로 생성)
- ✅ 화자 태그 사용 방식 학습 (예: "#Person1#은... #Person2#는...")
- ⚠️ 프롬프트 길이 증가 → 대화가 긴 경우 트런케이션 위험

**3-shot 효과**:
- ⚠️ 과도한 few-shot은 오히려 성능 저하 가능
- 프롬프트가 너무 길어져서 실제 대화 부분이 잘림
- **권장**: MBR 앙상블에서 0-shot, 1-shot, 3-shot을 혼합하여 다양성 확보

**실측 결과** (MBR 앙상블 기여도):
```
0-shot (dev_save, abstract):  48% 선택 ← 가장 자주 선택됨
1-shot (goldstyle):           17% 선택
3-shot (goldstyle_v2):        14% 선택
```
→ 단순한 0-shot이 가장 안정적이지만, Few-shot도 31% 기여

### 9-4. Qwen3 Thinking Mode 제어 (필수)

```python
# 학습 및 추론 모두에서 enable_thinking=False 필수
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    enable_thinking=False,      # <think> 블록 생성 방지
    add_generation_prompt=True  # 추론 시에만
)
```

---

## 10. 평가 파이프라인

### 10-1. 대회 공식 평가 방식

- **평가 지표**: MeCab 형태소 분석 기반 ROUGE-1/2/L F1
- **평가 식**: `Score = ROUGE-1-F1 + ROUGE-2-F1 + ROUGE-L-F1`
- **test.csv**: 대화 1개당 summary 3개 기준으로 개별 채점 후 종합

### 10-2. 로컬 평가 스크립트

```python
import mecab
from rouge import Rouge

def evaluate_rouge(predictions, references):
    m = mecab.MeCab()
    rouge = Rouge()
    
    preds_m = [" ".join(m.morphs(p)) if p.strip() else "빈요약" for p in predictions]
    refs_m = [" ".join(m.morphs(r)) if r.strip() else "빈요약" for r in references]
    
    scores = rouge.get_scores(preds_m, refs_m, avg=True)
    return {
        'rouge-1': scores['rouge-1']['f'],
        'rouge-2': scores['rouge-2']['f'],
        'rouge-l': scores['rouge-l']['f'],
        'total': sum(scores[k]['f'] for k in scores),
    }
```

### 10-3. 단계별 성능 목표

| 단계 | 방법 | 목표 ROUGE-1 |
|-----|-----|------------|
| Step 0 | Baseline SFT (전체 시퀀스) | 0.32 (현재) |
| Step 1 | Response-Only SFT | 0.56+ |
| Step 2 | Response-Only + LoRA r=32 | 0.57+ |
| Step 3 | Response-Only + SimPO 개선 | 0.58+ |
| Step 4 | + MBR 8개 앙상블 | 0.58~0.60+ |
| Step 5 | + 프롬프트 최적화 | 0.60+ |
| **최종 목표** | **모델 앙상블 + MBR** | **0.62+** |

---

## 11. 실험 로드맵 및 우선순위

### 11-1. 최종 운영 우선순위 (RTX 3090 환경)

**[우선순위 1] Qwen3-14B (R=32) 기반 Response-Only SFT 베이스라인 구축**
- `DataCollatorForCompletionOnlyLM` 적용하여 Response-Only Loss 구현
- LoRA r=32, alpha=32 설정
- `enable_thinking=False`로 Thinking 모드 비활성화
- Flash Attention 2 + Unsloth 최적화 조합
- 하이퍼파라미터: LR 2e-4, Epochs 3, Effective Batch 32 (1×32)
- 예상 ROUGE-1: 0.56+

**[우선순위 2] SFT 모델 결과물을 활용한 SimPO 데이터셋 구성**
- Llama 4 Maverick API로 train.csv 데이터 정제 (선택적)
- SimPO 학습을 위한 선호도(Chosen/Rejected) 데이터 생성
- On-policy Rejected 생성 (Temperature=0.7, num_return_sequences=4)
- ROUGE 기반 chosen/rejected 페어링

**[우선순위 3] SimPO 정렬 학습을 통해 요약문의 품질 고도화 및 최종 제출 파일 생성**
- SimPO (beta=5.0, gamma=1.0, lr=3e-7, r=64, 1 에폭)
- 간결성과 사실 관계 정확도 향상
- MBR 앙상블 디코딩 (n=8) 적용
- 예상 ROUGE-1: 0.58~0.60+

### 11-2. 확장 실험 (선택적, 시간 여유 시)

**[P4] 모델 교체 실험**
- Solar 10.7B SFT → ROUGE 비교
- Ministral3-14B SFT → ROUGE 비교
- 모델별 최고 성능 선정

**[P5] 프롬프트 변형 A/B 실험**
- 주제(topic) 활용 여부 검증
- 예상 효과: ±0.005

**[P6] LoRA Rank 확장 탐색**
- r=32 vs r=64 비교
- 메모리-성능 트레이드오프 측정

**[P7] 멀티모델 앙상블**
- Solar 10.7B + Qwen3-14B + Ministral3-14B
- 각 모델의 MBR 후보를 합쳐 최종 MBR
- 예상 ROUGE-1: 단일 모델 최고 대비 +0.01~0.03

**[P8] 데이터 품질 정제**
- 백트랜스레이션 데이터 ROUGE 기준 필터링
- 저품질 증강 데이터 제거 후 성능 변화 측정

### 11-3. 실험 추적 및 관리

**WandB 실험 명명 규칙**
```
{model}_{lora_r}_{loss_type}_{epoch}_{특징}
예: qwen3_14b_r32_response_only_3ep
    qwen3_14b_r32_simpo_beta5.0_1ep
    qwen3_14b_r32_mbr_n8_final
```

**디렉토리 구조**
```
outputs/
├── exp01_baseline_sft/
├── exp02_response_only/
├── exp03_lora_r32/
├── exp04_simpo_improved/
└── predictions/
    ├── exp01_dev_rouge.json
    └── exp02_dev_rouge.json
```

### 11-4. 실패한 시도들 (교훈 정리) — **실전 경험**

실제 프로젝트에서 시도했으나 효과가 없거나 실패한 방법들을 문서화합니다:

| 시도 | 결과 | 원인 | 교훈 |
|------|------|------|------|
| **Beam Search (k=4)** | 지원 안 됨 | Unsloth 4-bit 모델 한계 | 4-bit 양자화 모델은 일부 고급 디코딩 기법 미지원 |
| **Sampling + Reranking** | 성능 향상 없음 | Greedy 대비 차이 미미 | 단순한 Greedy decoding도 Response-Only Loss와 결합 시 충분히 효과적 |
| **max_new_tokens 줄이기 (100)** | ROUGE 하락 | 요약이 중간에 잘림 | 평균 요약 길이(85자≈50토큰) + 안전 마진 고려 필요 |
| **SimPO 적용** | 성능 저하 (0.32 기준) | Chosen/Rejected 품질 차이 부족 | SimPO는 이론적으로 우수하나 실전 적용 시 데이터 준비가 핵심 (섹션 5 참조) |
| **Qwen3-32B 사용** | 14B보다 낮음 (0.5433) | 과적합 또는 한국어 데이터 부족 | 큰 모델이 항상 좋은 것은 아님. 14B가 최적점 |
| **Few-shot 3개 이상** | 성능 저하 | 프롬프트 길이 증가로 대화 잘림 | 1-shot이 최적, 3-shot까지만 허용 |

**핵심 교훈**:
1. ✅ **단순함이 최선**: Greedy decoding + Response-Only Loss가 가장 효과적
2. ✅ **모델 크기 != 성능**: Qwen3-14B가 32B보다 우수
3. ✅ **데이터 품질 > 알고리즘**: SimPO는 좋은 Chosen/Rejected 페어가 핵심
4. ✅ **프롬프트 간결성**: 긴 프롬프트는 오히려 역효과

### 11-5. 실제 학습 시간 (실측) — **예산 계획용**

**GPU별 학습 시간** (Qwen3-14B, LoRA r=32, 3 에폭 기준):

| GPU | 메모리 | 배치 설정 | 에폭당 시간 | 총 학습 시간 | 비고 |
|-----|-------|----------|-----------|-----------|------|
| **RTX 3090** | 24GB | 1×32 | ~90분 | **4.5시간** | 프로젝트 실측 |
| RTX A6000 | 48GB | 4×8 | ~50분 | 2.5시간 | 추정 |
| A100 | 80GB | 8×4 | ~30분 | 1.5시간 | 추정 |

**추론 시간** (dev 499개, RTX 3090 기준):
- 단일 프롬프트 Greedy: **~5분**
- MBR 8 프롬프트: **~40분** (8배)
- Test 499개 추론: 단일 ~5분, MBR ~40분

**전체 파이프라인 시간 (RTX 3090)**:
```
환경 준비:          30분
SFT 학습:           4.5시간
Dev 평가:           5분
프롬프트 8개 추론:   ~6시간 (8×40분)
MBR 앙상블:         10분
Test 제출 파일:     40분
────────────────────────
총 소요 시간:       약 12시간
```

**비용 절감 팁**:
- Dev 평가는 단일 프롬프트로만 수행 (5분)
- MBR 앙상블은 최종 제출용으로만 사용
- 실험 단계에서는 Greedy만 사용하여 빠르게 반복

---

## 실전 구현 체크리스트

이 섹션은 실제 프로젝트를 진행할 때 단계별로 확인해야 할 체크리스트입니다. 각 Phase를 완료하면서 체크해 나가세요.

### Phase 1: 환경 준비

- [ ] **Unsloth 설치 및 CUDA 환경 확인**
  ```bash
  pip install unsloth
  python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
  ```

- [ ] **데이터 경로 설정 및 확인**
  ```python
  import pandas as pd
  train_df = pd.read_csv("data/train.csv")  # 17,341개 확인
  dev_df = pd.read_csv("data/dev.csv")      # 499개 확인
  test_df = pd.read_csv("data/test.csv")    # 499개 확인
  ```

- [ ] **Mecab 형태소 분석기 설치 및 테스트**
  ```bash
  pip install mecab-python3
  python -c "import mecab; m = mecab.MeCab(); print(m.morphs('테스트'))"
  ```

- [ ] **시드 고정 (재현성 확보)**
  ```python
  import random, numpy as np, torch
  random.seed(3407)
  np.random.seed(3407)
  torch.manual_seed(3407)
  ```

### Phase 2: 모델 학습

- [ ] **Qwen3-14B 4-bit 로드 확인 (~8.5GB VRAM 사용)**
  ```python
  from unsloth import FastLanguageModel
  model, tokenizer = FastLanguageModel.from_pretrained(
      "unsloth/Qwen3-14B", max_seq_length=2048, load_in_4bit=True
  )
  # VRAM 사용량 확인: ~8.5GB
  ```

- [ ] **LoRA r=32, alpha=32 설정**
  ```python
  model = FastLanguageModel.get_peft_model(
      model, r=32, lora_alpha=32, ...
  )
  # 학습 가능 파라미터: ~1.28억 개 (전체의 0.86%)
  ```

- [ ] **ResponseOnlyDataCollator 구현 및 검증**
  - Response template 토큰 ID 확인: `"<|im_start|>assistant\n"`
  - 마스킹 비율 체크: 약 70-80% (프롬프트 부분)
  - 샘플 1개로 동작 테스트

- [ ] **학습 시작 (3 에폭, RTX 3090 기준 ~4.5시간)**
  - 하이퍼파라미터: LR 2e-4, Batch 1×32, Flash Attention 2
  - 학습 중 loss 감소 확인 (초기 ~2.0 → 최종 ~0.5)
  - 메모리 사용량 모니터링: ~10GB 이하 유지

- [ ] **LoRA 어댑터 저장**
  ```python
  model.save_pretrained("outputs/qwen3_r32_sft/lora_adapter")
  tokenizer.save_pretrained("outputs/qwen3_r32_sft/lora_adapter")
  ```

### Phase 3: 검증

- [ ] **Dev set 추론 (499개, ~5분)**
  - Greedy decoding 사용 (`do_sample=False`)
  - `enable_thinking=False` 확인
  - 후처리 함수 적용

- [ ] **Mecab ROUGE 계산**
  ```python
  from rouge import Rouge
  import mecab
  m = mecab.MeCab()
  rouge = Rouge()
  # ROUGE-1/2/L F1 계산
  ```

- [ ] **목표 달성 확인: ROUGE-1 > 0.56**
  - 0.56 이상: ✅ Response-Only Loss 효과 확인
  - 0.50~0.56: 프롬프트 또는 후처리 개선 필요
  - 0.50 이하: 설정 재확인 필요

- [ ] **후처리 함수 적용 확인**
  - 화자 태그 정규화: `#Person 1#` → `#PersonN#`
  - 특수 토큰 제거: `<think>`, `<|im_start|>` 등
  - ROUGE +0.005~0.01 향상 확인

### Phase 4: MBR 앙상블 (선택적, 최종 제출용)

- [ ] **8개 프롬프트 변형 정의**
  - dev_save (기본)
  - abstract (추상적)
  - goldstyle (1-shot)
  - goldstyle_v2 (3-shot)
  - + 4개 추가 변형

- [ ] **각 프롬프트로 dev/test 추론 (~6시간)**
  - 프롬프트별 추론 결과 저장
  - CSV 파일로 저장: `submission_{prompt_name}.csv`

- [ ] **MBR 알고리즘 적용 (~10분)**
  - Pairwise ROUGE-1 계산
  - 평균 유사도가 가장 높은 후보 선택
  - 모델 선택 빈도 확인

- [ ] **목표 달성 확인: ROUGE-1 > 0.57**
  - 0.5716 달성 시: ✅ MBR 효과 확인
  - 단일 모델과 차이 미미: 프롬프트 다양성 부족

### Phase 5: 제출

- [ ] **Test set 최종 추론**
  - MBR 앙상블 결과 사용 (또는 단일 최고 성능)
  - 499개 요약 생성

- [ ] **submission.csv 생성 및 검증**
  ```python
  submission = pd.DataFrame({
      "fname": test_df["fname"],
      "summary": final_predictions
  })
  submission.to_csv("submission.csv", index=False)
  ```

- [ ] **제출 파일 체크**
  - 컬럼 확인: `fname`, `summary` 2개
  - 행 수 확인: 499개
  - 화자 태그 포함 여부: ~95% 이상 포함
  - 빈 요약 여부: 없어야 함

- [ ] **최종 제출 전 재확인**
  - Dev set ROUGE: 0.56+ (단일) 또는 0.57+ (MBR)
  - Test set 요약 길이: 평균 60~100자
  - 샘플 10개 육안 검토: 자연스러움, 화자 태그, 완전한 문장

### 트러블슈팅 체크리스트

**ROUGE 점수가 낮을 때 (< 0.50)**:
- [ ] Response-Only Loss 적용 확인 (마스킹 비율 체크)
- [ ] `enable_thinking=False` 설정 확인
- [ ] 후처리 함수 적용 확인
- [ ] LoRA rank 확인 (r=32 권장)
- [ ] 학습 loss가 충분히 감소했는지 확인

**메모리 부족 오류 발생 시**:
- [ ] `max_seq_length` 줄이기 (2048 → 1536)
- [ ] `gradient_accumulation_steps` 늘리기 (32 → 64)
- [ ] Flash Attention 2 적용 확인
- [ ] Gradient Checkpointing 활성화 확인

**추론 속도가 너무 느릴 때**:
- [ ] 4-bit 양자화 적용 확인
- [ ] Unsloth `for_inference()` 호출 확인
- [ ] 배치 추론 사용 (여러 샘플 동시 처리)
- [ ] MBR은 최종 제출용으로만 사용

---

## 12. 참고문헌 및 출처

### 모델 공식 문서

1. **SOLAR-10.7B**: Kim et al., "SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling", arXiv:2312.15166 (2023)  
   HF: `upstage/SOLAR-10.7B-v1.0` | 라이선스: Apache 2.0

2. **Qwen3**: Qwen Team, "Qwen3 Technical Report", arXiv:2505.09388 (2025)  
   HF: `Qwen/Qwen3-14B` | 라이선스: Apache 2.0  
   한국어 특화 연구: "Making Qwen3 Think in Korean with RL", arXiv:2508.10355 (2025)

3. **Ministral 3**: Mistral AI, "Introducing Mistral 3" (2025년 12월)  
   HF: `mistralai/Ministral-3-14B-Instruct-2512` | 라이선스: Apache 2.0

4. **Llama 4 Maverick**: Meta, "Llama 4 Community Release" (2025년 4월)  
   HF: `meta-llama/Llama-4-Maverick-17B-128E-Instruct` | 라이선스: Llama 4 Community License

### 알고리즘 논문

5. **SimPO**: Meng et al., "SimPO: Simple Preference Optimization with a Reference-Free Reward", NeurIPS 2024, arXiv:2405.14734  
   GitHub: `princeton-nlp/SimPO` | 권장 beta=2.0~10.0, gamma=0.5~1.5, LR=3e-7~1e-6

6. **DPO**: Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", NeurIPS 2023

7. **ORPO**: Hong et al., "ORPO: Monolithic Preference Optimization without Reference Model", arXiv:2403.07691 (2024)

8. **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022

9. **QLoRA**: Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", NeurIPS 2023

10. **MBR Decoding**: Müller & Sennrich, "Understanding the Properties of Minimum Bayes Risk Decoding in NMT", ACL 2021

### 도구 및 라이브러리

11. **Unsloth**: https://github.com/unslothAI/unsloth (2× 속도, 70% 메모리 절약 LoRA 학습)
12. **TRL (HuggingFace)**: https://github.com/huggingface/trl (SFTTrainer, CPOTrainer, ORPOTrainer)
13. **MeCab-python3**: 한국어 형태소 분석기 (`pip install mecab-python3`)
14. **rouge 라이브러리**: ROUGE 계산 (`pip install rouge`)

---

## 부록: 빠른 시작 가이드 (Response-Only SFT)

```python
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from datasets import Dataset
import pandas as pd

# 1. 모델 로드
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3-14B",
    max_seq_length=2048,
    load_in_4bit=True,
)

# 2. LoRA 설정 (r=32 권장)
model = FastLanguageModel.get_peft_model(
    model, r=32, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth", random_state=42,
)

# 3. 데이터 준비
train_df = pd.read_csv("data/train.csv")
SYSTEM = "당신은 한국어 대화 요약 전문가입니다. 화자 태그를 유지하며 1~3문장으로 요약하세요."
USER_T = "아래 대화를 요약하세요. 화자 태그(#Person1# 등)를 유지하세요.\n\n{dialogue}"

def fmt(row):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_T.format(dialogue=row["dialogue"])},
        {"role": "assistant", "content": str(row["summary"])},
    ]
    return {"text": tokenizer.apply_chat_template(msgs, tokenize=False, enable_thinking=False)}

dataset = Dataset.from_pandas(train_df).map(fmt)

# 4. Response-Only Collator (핵심!)
collator = DataCollatorForCompletionOnlyLM(
    response_template="<|im_start|>assistant\n",
    tokenizer=tokenizer,
)

# 5. 학습 실행
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    data_collator=collator,
    processing_class=tokenizer,
    args=SFTConfig(
        output_dir="outputs/response_only_r32",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        optim="adamw_8bit",
        max_seq_length=2048,
        seed=42,
        report_to="none",
        save_strategy="epoch",
        save_total_limit=2,
    ),
)
trainer.train()
model.save_pretrained("outputs/response_only_r32/lora_adapter")
tokenizer.save_pretrained("outputs/response_only_r32/lora_adapter")
```

---

*본 문서는 첨부된 베이스라인 코드(`대화요약_Baseline_SFT.ipynb`, `대화요약_SimPO.ipynb`), 데이터셋(`train.csv`, `dev.csv`, `test.csv`) 분석 결과와 HuggingFace, arXiv 등 공식 문서 검색을 기반으로 작성되었습니다. 2026년 3월 20일 최종 전략(RTX 3090 환경, Qwen3-14B Primary, SimPO 파이프라인)을 반영하였습니다.*
