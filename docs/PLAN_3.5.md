# Qwen3.5-9B Response-Only SFT 적용 계획

작성일: 2026-03-27
기준 LB: 52.0083 (Qwen3-14B r4b + qa_style)

---

## 1. 모델 개요

| 항목 | Qwen3-14B (현재) | Qwen3.5-9B (신규) |
|------|-----------------|-----------------|
| 파라미터 | 14B | 9B |
| 아키텍처 | 순수 Attention | Gated DeltaNet + Sparse Attention (하이브리드) |
| 컨텍스트 | 32K | 262K |
| 4-bit VRAM | ~7-8 GB | ~4.5 GB |
| Thinking 모드 | ✅ | ✅ |
| 라이센스 | Apache 2.0 | Apache 2.0 |
| Unsloth BNB 4-bit | `unsloth/qwen3-14b-unsloth-bnb-4bit` 존재 | **미출시** (GGUF만 존재) |
| Unsloth 지원 여부 | ✅ | ✅ (transformers ≥ 5.2.0 필요) |

현재 환경: transformers 5.3.0, unsloth 2026.3.8 → **호환 가능**

---

## 2. 장단점 분석

### 장점
- **VRAM 여유**: 4.5GB (vs 14B 7-8GB) → batch_size, grad_accum 확대 가능
- **학습 속도**: 14B 대비 약 40~50% 빠름 → sweep 실험 1회 ~1h
- **최신 아키텍처**: Gated DeltaNet + Sparse MoE 하이브리드 → 이론상 표현력 향상
- **긴 컨텍스트**: 262K → 긴 대화에도 안정적
- **데이터 증강 활용**: VRAM 여유로 `data_aug/train.csv` (증강 포함) 사용 가능

### 단점 / 리스크
- **사전 양자화 버전 없음**: `Qwen/Qwen3.5-9B` 로드 시 fp16 다운로드 후 온-더-플라이 양자화 → 초기 로드 5~10분 추가
- **14B보다 작음**: 같은 학습 조건에서 성능 열위 가능성 존재
- **미검증**: 한국어 대화 요약 벤치마크 없음 (MMLU-ProX 29개 언어 평균만 공개)
- **하이브리드 아키텍처**: LoRA target_modules 동일하나 DeltaNet 레이어 학습 효과 미검증

---

## 3. 데이터 전처리 (`src/data/preprocess.py`)

### 3.1 데이터 현황

| 파일 | 행 수 | 설명 |
|------|-------|------|
| `data/train.csv` | ~12,457 | 기본 학습 데이터 |
| `data_aug/train.csv` | **17,341** | 증강 포함 통합본 |
| `data_aug/train_aug_back_translation.csv` | 2,384 | 역번역 증강 |
| `data_aug/train_aug_eda.csv` | 2,500 | EDA 증강 |
| `data/dev.csv` | ~499 | 검증 |
| `data/test.csv` | ~499 | 제출용 |

Qwen3.5-9B의 VRAM 여유를 활용해 **`data_aug/train.csv` (17,341행) 사용 권장**.

---

### 3.2 전처리 파이프라인

`src/data/preprocess.py`의 함수들을 순서대로 적용:

```
[CSV 로드]
    ↓
[clean_text()]        ← use_cleaning: true (config.yaml)
    ↓
[filter_by_length()]  ← use_length_filter: true (config.yaml)
    ↓
[chat template 포맷]  ← qa_style system + user 프롬프트
    ↓
[Response-Only 마스킹] ← DatasetForCausalLM / ResponseOnlyDataCollator
```

#### Step 1 — `clean_text()`: 노이즈 제거

```python
from src.data.preprocess import clean_text

# dialogue와 summary 양쪽에 모두 적용
df["dialogue"] = df["dialogue"].apply(clean_text)
df["summary"]  = df["summary"].apply(clean_text)
```

제거 항목:
- 단독 자음/모음 (`ㅋㅋ`, `ㅠㅠ` 등)
- 빈 괄호 (`()`, `[]`, `{}`)
- 3회 이상 반복 특수기호 (`!!!` → `!`)
- 중복 공백

#### Step 2 — `filter_by_length()`: 이상치 제거

```python
from src.data.preprocess import filter_by_length

df = filter_by_length(
    df,
    dialogue_max=830,   # 대화 최대 830자 (train max 2,168 → 이상치 제거)
    summary_min=50,     # 요약 최소 50자
    summary_max=250,    # 요약 최대 250자
)
# 예시: 17,341 → ~16,800행 (약 3% 제거)
```

#### Step 3 — Chat Template 포맷

```python
QA_SYSTEM = (
    "당신은 한국어 대화 요약 전문가입니다.\n"
    "대화를 분석하여 화자들이 무엇을 논의하고 어떤 결정을 내렸는지 요약하세요.\n"
    "#Person1#, #Person2# 등의 화자 표기를 그대로 유지하세요.\n"
    "간결하게 1~2문장으로 작성하세요."
)
QA_USER = "이 대화에서 무슨 일이 일어났나요?\n\n{dialogue}"

def make_text(row, tokenizer):
    messages = [
        {"role": "system",    "content": QA_SYSTEM},
        {"role": "user",      "content": QA_USER.format(dialogue=row["dialogue"])},
        {"role": "assistant", "content": row["summary"]},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
```

#### Step 4 — Response-Only 마스킹

기존 `ResponseOnlyDataCollator` (run_qlora_sweep.py) 재사용:

```python
# assistant 응답 시작 토큰 ID 찾기 (Qwen3.5 chat template 기준 검증 필요)
response_template = "<|im_start|>assistant\n"
response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)

collator = ResponseOnlyDataCollator(
    tokenizer=tokenizer,
    response_template_ids=response_template_ids,
    max_length=MAX_SEQ_LENGTH,  # 2048
)
```

> **주의**: Qwen3.5 tokenizer로 `response_template_ids`를 재검증해야 합니다.
> Qwen3와 다를 수 있으므로 아래처럼 확인:
> ```python
> print(tokenizer.decode(response_template_ids))
> # 예상 출력: <|im_start|>assistant\n
> ```

---

### 3.3 TTA (선택: 추론 시 앙상블)

`reverse_utterances()` + `apply_tta()` 를 추론 시 활용 가능:

```python
from src.data.preprocess import apply_tta

# 각 대화에 대해 [원본, 역순] 2가지 변형으로 추론 후 평균
tta_variants = apply_tta(dialogues, n_ways=2)
```

단, ROUGE는 n-gram 매칭이라 역순 입력이 오히려 해가 될 수 있음 → dev에서 검증 후 결정.

---

## 4. 모델 로드

### 3.1 모델 로드 방법

pre-quantized 버전이 없으므로 두 가지 옵션:

**Option A — Unsloth FastModel (권장)**
```python
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="Qwen/Qwen3.5-9B",
    max_seq_length=2048,
    load_in_4bit=True,
    device_map={"": 0},
)
```
- Unsloth가 자동으로 qwen3_5 패치 적용
- 첫 실행 시 fp16 모델 다운로드 (~18GB) 후 4bit 변환

**Option B — 표준 transformers + BitsAndBytesConfig**
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-9B",
    quantization_config=bnb,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
)
```

> **Unsloth pre-quantized 출시 대기**: `unsloth/Qwen3.5-9B-unsloth-bnb-4bit` 출시 시 가장 빠름.
> HuggingFace 검색: https://huggingface.co/unsloth

### 3.2 Response Template 확인

Qwen3.5의 chat template은 Qwen3와 동일한 패턴 사용:

```python
tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
```

Response-only 마스킹용 template ID 확인:
```python
# run_qlora_sweep.py 에서 response_template 재검증 필요
response_template = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
```

---

## 5. 학습 스크립트 수정 사항

기존 `LLM/response_only_SFT/run_qlora_sweep.py` 대비 변경 필요 항목:

### 4.1 최소 변경 (run_qlora_sweep_q35.py 신규 생성)

```python
# 변경 전
MODEL_NAME = "unsloth/Qwen3-14B"
MAX_SEQ_LENGTH = 2048
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 32

# 변경 후
MODEL_NAME = "Qwen/Qwen3.5-9B"   # pre-quantized 미존재 → 직접 로드
MAX_SEQ_LENGTH = 2048
PER_DEVICE_BATCH = 2              # VRAM 여유로 batch 확대 가능
GRAD_ACCUM = 16                   # 동일 effective batch (32) 유지
```

### 4.2 모델 로드 변경

```python
# 기존 (Qwen3-14B pre-quantized)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

# 신규 (Qwen3.5-9B, FastModel 사용)
from unsloth import FastModel
model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    device_map={"": 0},
)
model = FastModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
)
```

### 4.3 데이터 증강 활용 (선택)

VRAM 여유로 더 많은 학습 데이터 사용 가능:
```python
# 기존
DATA_PATH = "LLM/response_only_SFT/data/train.csv"  # 12,457개

# 증강 데이터 포함 시
DATA_PATH = "data_aug/train.csv"   # 증강 포함 버전 사용 가능
```

---

## 6. 권장 실험 구성

| 실험 | r | alpha | lr | 데이터 | 예상 학습 시간 |
|------|---|-------|----|--------|-------------|
| q35_A | 32 | 64 | 1e-4 | 기본 | ~1h |
| q35_B | 64 | 128 | 2e-4 | 기본 | ~1h |
| q35_C | 32 | 64 | 1e-4 | data_aug | ~1.5h |

1 epoch, qa_style 프롬프트로 dev ROUGE 비교 후 최고 설정 test 추론.

---

## 7. MBR 디코딩 전략 (`LLM/prompts/`)

### 7.1 MBR 개요

MBR(Minimum Bayes Risk) 디코딩: 여러 프롬프트로 생성한 후보 요약 중,
**다른 모든 후보와의 ROUGE 기대값이 최대인 요약**을 최종 선택.

```
y* = argmax_y (1/|H| × Σ_{y'∈H} ROUGE(y, y'))
```

실측 성능 (Qwen3-14B 기준):
- Greedy 단일: ROUGE-1 0.5641
- MBR 7종: ROUGE-1 0.5716 (+0.0075, +1.3%)

**단, LB 결과**: MBR 51.7849 < qa_style 단독 52.0083
→ 이유: 품질 낮은 프롬프트가 MBR 투표에서 노이즈로 작용

---

### 7.2 활성 프롬프트 변형 7종

| 이름 | LB (단독) | 설명 |
|------|----------|------|
| `qa_style` | **52.0083** | "이 대화에서 무슨 일이?" — 현재 최고 |
| `observer` | 50.8513 | 제3자 관찰자 시점 |
| `gold_mimic` | 50.6548 | Gold 패턴 규칙 명시 |
| `length_constrained` | 50.5627 | 50~150자 길이 제약 |
| `narrative` | — | 서술형 |
| `topic` | — | Topic 힌트 활용 |
| `base` | — | 기본 프롬프트 |

비활성 (주석처리): `abstract`, `oneshot`, `threeshot`, `base_copy`
→ 실험 결과 단독 성능 낮아 MBR 노이즈만 추가

---

### 7.3 MBR 추론 코드

```python
from LLM.prompts.mbr_prompts import get_all_prompt_variants, create_messages
from LLM.prompts.mbr_decoding import apply_mbr_to_dataset

# 1. 각 프롬프트로 추론
all_predictions = {}
for name, variant in get_all_prompt_variants().items():
    preds = []
    for dialogue in dialogues:
        messages = create_messages(name, dialogue)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        # ... generate ...
        preds.append(pred)
    all_predictions[name] = preds

# 2. MBR로 최적 요약 선택
final_preds = apply_mbr_to_dataset(
    test_df,
    all_predictions,
    use_mecab=True,       # MeCab 형태소 분석 (대회 평가 기준)
    metric="rouge-1",     # 선택 기준 메트릭
)

# 3. 멀티메트릭 MBR (R1+R2+RL 종합)
from LLM.prompts.mbr_decoding import mbr_multi_metric
# apply_mbr_to_dataset의 내부 호출을 mbr_multi_metric으로 교체 가능

# 4. 가중치 MBR (qa_style에 높은 가중치)
from LLM.prompts.mbr_decoding import mbr_with_weights
weights = {"qa_style": 2.0, "observer": 1.2, "gold_mimic": 1.0, ...}
```

---

### 7.4 Qwen3.5-9B MBR 전략

Qwen3.5-9B는 Qwen3-14B와 **아키텍처가 달라** 프롬프트별 출력 분포가 다를 수 있음.

**권장 실험 순서**:

1. `qa_style` 단독 → dev ROUGE 측정 (기준선)
2. `qa_style` + `observer` 2종 MBR → 개선 여부 확인
3. 활성 7종 전체 MBR → 단독 대비 비교

```bash
# qa_style 단독
cd LLM && python run_prompts.py \
  --model_path response_only_SFT/outputs/q35_best/lora_adapter \
  --mode inference \
  --prompt_variant qa_style \
  --test_file test.csv \
  --output_file ../prediction/q35_qa_test.csv

# MBR (7종 전체)
python run_prompts.py \
  --model_path response_only_SFT/outputs/q35_best/lora_adapter \
  --mode inference \
  --use_mbr \
  --test_file test.csv \
  --output_file ../prediction/q35_mbr_test.csv
```

**MBR가 도움이 되는 조건**:
- 여러 프롬프트의 단독 성능 편차가 작을수록 MBR 효과 큼
- qa_style이 압도적으로 높으면 단독이 유리

---

## 8. 최종 추론 및 앙상블

```bash
# dev 검증
cd LLM/response_only_SFT
python run_qlora_sweep_q35.py

# test 추론 (최고 체크포인트 자동 선택)
python run_test_inference_q35.py

# Qwen3-14B(52.0) + Qwen3.5-9B 앙상블
python src/ensemble_cli.py merge \
  --inputs prediction/qwen_test_qa_style_best.csv \
           prediction/q35_test_best.csv \
  --output prediction/ensemble_q3_q35.csv \
  --oof-scores 52.0083 <q35_dev_score>
```

---

## 8. 실행 순서 체크리스트

```
[ ] Unsloth pre-quantized 버전 출시 여부 확인
    → https://huggingface.co/unsloth/Qwen3.5-9B-unsloth-bnb-4bit

[ ] 데이터 전처리 검증
    [ ] clean_text() 적용 후 샘플 확인 (화자 태그 #Person1# 보존 여부)
    [ ] filter_by_length() 결과 확인 (몇 행 제거되는지)
    [ ] data_aug/train.csv vs data/train.csv 행 수 비교 후 사용 데이터 결정
    [ ] Qwen3.5 tokenizer로 response_template_ids 재검증
        python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-9B'); print(t.decode(t.encode('<|im_start|>assistant\n', add_special_tokens=False)))"

[ ] LLM/response_only_SFT/run_qlora_sweep_q35.py 생성 (기존 스크립트 복사 + 수정)
    [ ] MODEL_NAME = "Qwen/Qwen3.5-9B"
    [ ] PER_DEVICE_BATCH = 2, GRAD_ACCUM = 16
    [ ] 전처리 파이프라인 (clean_text + filter_by_length) 적용
    [ ] data_aug/train.csv 경로 설정

[ ] q35_A 실험 1회 실행 → dev ROUGE 확인
[ ] 결과 비교: q35 dev vs r4b dev(52.0)
[ ] 개선 시: 추가 실험 (q35_B, q35_C)

[ ] MBR 전략 검증
    [ ] qa_style 단독 dev ROUGE 측정 (기준선)
    [ ] qa_style + observer 2종 MBR → dev 비교
    [ ] 7종 전체 MBR → 단독 대비 비교
    [ ] 최적 방식(단독 vs MBR) 결정 후 test 추론

[ ] 최종: Qwen3-14B(52.0) + Qwen3.5-9B 앙상블로 제출
    [ ] --oof-scores에 dev ROUGE 점수 입력
```

---

## 9. 기대 효과

- **독립적 예측**: Qwen3-14B(순수 attention)과 Qwen3.5-9B(하이브리드)는 아키텍처가 달라 앙상블 다양성 확보
- **VRAM 여유**: 두 모델을 순차 실행해도 OOM 없이 진행 가능
- **빠른 실험**: 9B라 한 번 sweep에 ~2~3h → 결과 빠르게 확인 가능
- **최악 시나리오**: q35 단독 성능이 낮아도 앙상블 효과로 현재 52.0 이상 기대
