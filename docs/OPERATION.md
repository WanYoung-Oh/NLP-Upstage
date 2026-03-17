# 파이프라인 운영 가이드

> 한국어 대화 요약 파이프라인의 설치·학습·추론·평가·앙상블 전 과정을 다룹니다.

---

## 목차

1. [환경 설정](#1-환경-설정)
2. [디렉토리 구조](#2-디렉토리-구조)
3. [학습 (Training)](#3-학습-training)
4. [추론 (Inference)](#4-추론-inference)
5. [평가 (Evaluation)](#5-평가-evaluation)
6. [앙상블 (Ensemble)](#6-앙상블-ensemble)
7. [체크포인트 관리](#7-체크포인트-관리)
8. [WandB 모니터링](#8-wandb-모니터링)
9. [설정 파일 레퍼런스](#9-설정-파일-레퍼런스)
10. [모델별 운영 가이드](#10-모델별-운영-가이드)
11. [실험 자동화 스크립트](#11-실험-자동화-스크립트)
12. [단위 테스트](#12-단위-테스트)
13. [트러블슈팅](#13-트러블슈팅)
    - 13-7. [Causal LM batch_size mismatch 오류](#13-7-causal-lm-학습-시-batch_size-mismatch-오류) (신규)

---

## 1. 환경 설정

### 1-1. 패키지 설치

```bash
cd /data/ephemeral/home/NLP
pip install -r requirements.txt
```

### 1-2. 환경 변수

`.env` 파일에 아래 값을 입력합니다. `train.py`는 시작 시 `load_dotenv()`로 자동 로드합니다.

```dotenv
WANDB_ENTITY=your_wandb_entity
WANDB_PROJECT=your_project_name
UPSTAGE_API_KEY=your_upstage_key   # Solar API 사용 시
```

### 1-3. 데이터 확인

```
data/
├── train.csv   # 12,457건 (fname, dialogue, summary)
├── dev.csv     #    499건 (fname, dialogue, summary)
└── test.csv    #    499건 (fname, dialogue — summary 없음)
```

---

## 2. 디렉토리 구조

```
NLP/
├── conf/                    # Hydra 설정 파일
│   ├── config.yaml          # 최상위 설정 (defaults + general + tokenizer + data + metrics)
│   ├── model/               # 모델별 config
│   ├── training/            # 학습 설정
│   └── inference/           # 추론 설정
│       ├── beam4.yaml
│       ├── beam8.yaml
│       ├── mbr.yaml
│       ├── tta.yaml
│       ├── solar_api.yaml
│       └── zero_shot_solar.yaml   # Solar zero-shot 추론 (신규)
├── src/
│   ├── train.py             # 학습 진입점
│   ├── inference.py         # 추론 진입점
│   ├── ensemble.py          # 앙상블 / GroupKFold OOF / MBRDecoder
│   ├── data/
│   │   ├── preprocess.py    # Preprocess, clean_text, filter_by_length,
│   │   │                    # DatasetForSeq2Seq, DatasetForCausalLM, DatasetForInference,
│   │   │                    # reverse_utterances, apply_tta
│   │   ├── augment.py       # EDA / back-translation 증강 핵심 로직
│   │   └── run_augment.py   # 증강 CLI 진입점
│   ├── models/
│   │   └── summarizer.py    # 모델 로드 (BART / T5 / CausalLM 분기)
│   └── utils/
│       ├── device.py        # 디바이스 자동 감지 (CUDA > MPS > CPU)
│       ├── metrics.py       # ROUGE 평가 (단일/다중 정답, korouge-score 지원)
│       └── postprocess.py   # 후처리 파이프라인 (5단계)
├── scripts/
│   ├── evaluate_on_dev.py   # dev ROUGE 평가 (후처리 전/후, beam4/8/MBR/TTA 비교)
│   └── run_all_experiments.sh  # Phase 1~5 전체 실험 자동화
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py     # 43개 단위 테스트 (GPU 불필요)
├── data/                    # 원본 CSV
├── data_aug/                # EDA/역번역 증강 결과 (run_augment.py 실행 후 생성)
│   ├── train.csv            #   원본 + 증강 합산
│   ├── train_aug_eda.csv    #   EDA 증강 결과만 (참고용)
│   ├── dev.csv              #   원본에서 복사
│   └── test.csv             #   원본에서 복사
├── checkpoints/             # 학습 체크포인트 (yymmdd_run_NNN/epoch##_score)
├── prediction/              # 추론 결과 CSV
├── logs/                    # run_all_experiments.sh 실행 로그
├── outputs/                 # Hydra 실행 로그
└── docs/                    # 프로젝트 문서
```

---

## 3. 학습 (Training)

### 3-0. 데이터 증강 (학습 전 선택 실행)

데이터 증강은 **학습 전 별도로 실행**해야 하는 선택적 전처리 단계입니다.
외부 데이터셋 없이 기존 `train.csv`를 변형해 새 샘플을 생성합니다.

| 방법 | 방식 | 외부 의존 |
|------|------|----------|
| `back_translation` | dialogue를 ko→en→ko 역번역 | Google Translate (인터넷) |
| `eda` | dialogue에서 단어 무작위 삭제 | 없음 (완전 오프라인) |

`src/data/run_augment.py`로 실행합니다. 증강 결과와 원본을 합산해 `output_dir`에 저장하고, `dev.csv`·`test.csv`도 자동으로 복사합니다.

```bash
# EDA만
python src/data/run_augment.py --method eda

# 역번역만
python src/data/run_augment.py --method back_translation

# 둘 다 적용 (train_aug_eda.csv + train_aug_back_translation.csv 모두 합산)
python src/data/run_augment.py --method all

# 옵션 조정
python src/data/run_augment.py --method all \
    --max_samples 1000        \  # 증강할 최대 샘플 수 (기본: 전체)
    --rouge_threshold 0.4     \  # ROUGE-L 유사도 최솟값 (기본: 0.3)
    --data_dir data           \  # 원본 CSV 디렉토리 (기본: data)
    --output_dir data_aug        # 출력 디렉토리 (기본: data_aug)
```

실행 완료 후 `data_aug/` 구조:

```
data_aug/
├── train.csv               # 원본 + 증강 합산
├── train_aug_eda.csv        # EDA 증강 결과만 (참고용)
├── train_aug_back_translation.csv   # 역번역 증강 결과만 (참고용)
├── dev.csv                 # 원본에서 복사
└── test.csv                # 원본에서 복사
```

이후 학습:

```bash
python src/train.py general.data_path=data_aug
```

---

### 3-1. 기본 실행

```bash
# KoBART + baseline 설정 (기본값)
python src/train.py

# 모델 변경
python src/train.py model=kot5 training=t5
python src/train.py model=kobart_v2
python src/train.py model=pko_t5 training=t5
python src/train.py model=solar_qlora training=qlora

# 학습 설정 변경
python src/train.py training=full          # 풀 학습 (lr=3e-5, epoch=50)
python src/train.py training=mps           # Apple Silicon 메모리 절약 모드
```

### 3-2. 하이퍼파라미터 직접 override

```bash
python src/train.py training.learning_rate=3e-5
python src/train.py training.num_train_epochs=30
python src/train.py training.per_device_train_batch_size=8
python src/train.py training.learning_rate=5e-5 training.num_train_epochs=30
```

### 3-3. 데이터 전처리 플래그 (Phase 3+)

`conf/config.yaml`의 `data` 섹션으로 전처리 옵션을 제어합니다.

```yaml
data:
  use_cleaning: false        # true: clean_text() 적용 — 단독 자음·빈 괄호·반복 특수기호 제거
  use_length_filter: false   # true: filter_by_length() 적용 — 이상치 길이 샘플 제거
```

CLI로 override:

```bash
# 클리닝만 활성화
python src/train.py data.use_cleaning=true

# 클리닝 + 길이 필터 둘 다 활성화
python src/train.py model=kot5 training=full data.use_cleaning=true data.use_length_filter=true
```

- `use_cleaning`: `dialogue`와 `summary` 컬럼 모두에 `clean_text()` 적용 (학습 데이터 한정)
- `use_length_filter`: `filter_by_length()` 적용 — dialogue > 1500자 또는 summary < 5자 / > 250자 샘플 제거 (학습 데이터 한정)
- 검증 데이터에는 클리닝만 적용, 길이 필터는 적용하지 않음 (일관성 유지)

### 3-4. Hydra Sweep (여러 설정 순차 실행)

```bash
python src/train.py -m model=kobart,kot5 training=baseline
python src/train.py -m training.learning_rate=1e-5,3e-5,5e-5
```

### 3-5. 최종 제출용 Train+Dev 합산 학습 (선택)

모든 실험·모델 선택이 끝난 뒤, 시간이 남을 때 시도하는 선택적 전략입니다.

**흐름**:
```
[실험] train.csv → 학습 / dev.csv → 검증 → best epoch 확인 (예: epoch 6)
[제출] train.csv + dev.csv → use_all_data=true로 epoch 6만큼 재학습 → inference
```

dev.csv의 499건(전체의 약 4%)을 추가 학습에 활용하지만, 개선 폭은 미미할 수 있습니다.
앙상블·MBR·TTA 등 다른 전략이 ROUGE 향상에 더 효과적입니다.

```bash
# best epoch이 6이었을 경우
python src/train.py training.use_all_data=true training.num_train_epochs=6
```

- `train.csv` + `dev.csv` 전체를 학습 데이터로 사용
- **eval 자동 비활성화** — dev가 학습 데이터에 포함되므로 eval을 돌리면 지표가 오염됨. `do_eval=False`·`eval_strategy="no"`·Early stopping·BestCheckpointCallback이 모두 자동으로 꺼짐
- `num_train_epochs`는 **직전 dev 검증에서 확인한 best epoch 수**로 직접 지정할 것

### 3-6. 평가 메트릭

| WandB 키 | 설명 |
|----------|------|
| `eval/rouge_1_f1` | ROUGE-1 F1 (0~1) |
| `eval/rouge_2_f1` | ROUGE-2 F1 (0~1) |
| `eval/rouge_l_f1` | ROUGE-L F1 (0~1) |
| `eval/rouge_combined` | R1+R2+RL 합산, 체크포인트 선택 기준 (최대 3.0) |

> 로컬 dev.csv는 정답 1개 기준. 대회 채점은 정답 3개 기준이라 로컬 점수보다 높게 나올 수 있습니다.

### 3-7. 한국어 ROUGE 활성화

`conf/config.yaml`에서 설정합니다.

```yaml
metrics:
  use_korouge: true    # Phase 3+ 권장. Java 불필요, korouge-score 사용
```

또는 CLI override:

```bash
python src/train.py metrics.use_korouge=true
```

### 3-8. 학습 출력물

- **체크포인트**: `checkpoints/{yymmdd_run_NNN}/epoch{##}_{rouge_combined:.4f}/`
  - rouge_combined 상위 3개만 유지 (`BestCheckpointCallback`)
  - 예: `checkpoints/260313_run_001/epoch06_0.7498/`
- **Hydra 로그**: `outputs/{date}/{time}/` (`.hydra/config.yaml` 포함)
- **WandB**: entity/project는 `.env`에서 읽어 자동 업로드

---

## 4. 추론 (Inference)

### 4-1. Beam Search (기본)

```bash
# Beam4 (기본)
python src/inference.py inference.ckt_path=checkpoints/260313_run_001/epoch06_0.7498

# Beam8 (length_penalty=1.2, 더 긴 탐색)
python src/inference.py inference=beam8 inference.ckt_path=checkpoints/...
```

### 4-2. MBR Decoding

N개 샘플을 생성한 뒤 ROUGE-L 기준으로 최적 문장을 선택합니다.

```bash
python src/inference.py inference=mbr inference.ckt_path=checkpoints/...
```

- `n_samples=10`, `temperature=0.9`, `top_p=0.95`
- Beam Search보다 느리지만 최종 제출 후보로 적합

### 4-3. TTA (Test-Time Augmentation)

발화 순서를 역전한 변형본으로 추론 후 MBRDecoder로 최적 후보를 선택합니다.

```bash
python src/inference.py inference=tta inference.ckt_path=checkpoints/...
```

- `n_tta_ways=2` (원본 + 발화역전) → 2개 후보 중 ROUGE-L 최고 선택
- `n_tta_ways`를 늘릴수록 후보가 많아지나 속도 감소

```bash
# n_tta_ways 직접 override
python src/inference.py inference=tta inference.n_tta_ways=3 inference.ckt_path=checkpoints/...
```

### 4-4. 동적 max_new_tokens

대회 규칙(대화 길이의 20% 이내)에 맞춘 동적 길이 제한:

```bash
python src/inference.py inference.max_length_ratio=0.2 inference.ckt_path=checkpoints/...
```

- `max_length_ratio=0.0` (기본): 고정 `generate_max_length=100` 사용
- `max_length_ratio=0.2`: 입력 토큰 수의 20% (최소 30토큰)

### 4-5. Solar API 추론

```bash
# Few-shot (기본)
python src/inference.py inference=solar_api

# Zero-shot
python src/inference.py inference=zero_shot_solar
```

`conf/inference/solar_api.yaml` 기본 설정:

```yaml
inference_type: solar_api     # 명시적 분기 플래그
model_name: solar-pro
temperature: 0.1
top_p: 0.9
max_tokens: 150
prompt_style: few_shot        # zero_shot | few_shot
n_few_shot: 3
use_bm25: true                # BM25 기반 동적 few-shot 예제 선택
rate_limit_rpm: 100
```

`conf/inference/zero_shot_solar.yaml` (신규):

```yaml
# beam4 기본값 상속 후 Solar zero-shot 항목 오버라이드
inference_type: "solar_api"
model_name: "solar-pro"
temperature: 0.1
top_p: 0.9
max_tokens: 150
prompt_style: "zero_shot"
n_few_shot: 0
use_bm25: false
rate_limit_rpm: 100
output_filename: "output_solar_zero_shot.csv"
```

> `UPSTAGE_API_KEY`가 `.env`에 설정되어 있어야 합니다.

### 4-6. 출력 결과

- 기본 저장 위치: `prediction/output.csv`
- 컬럼: `fname`, `summary`

```bash
# 파일명 변경
python src/inference.py inference.ckt_path=... inference.output_filename=output_beam8.csv
```

### 4-7. 재생성 필요 감지

추론 결과 후처리 시 10자 미만 요약문은 자동으로 경고가 출력됩니다:

```
[경고] 3/499개 요약문이 최소 길이 미달 (재생성 권장)
```

---

## 5. 평가 (Evaluation)

### 5-1. 로컬 단일 정답 평가 (학습 중 자동)

학습 중 `compute_metrics()`가 epoch마다 자동 호출됩니다. 결과는 WandB에 기록됩니다.

### 5-2. Dev 종합 평가 스크립트 (`scripts/evaluate_on_dev.py`)

PRD Phase 5 기준의 포괄적 평가를 한 번에 수행합니다.

```bash
# 기본 (best checkpoint 자동 탐색, beam4 후처리 전/후 비교)
python scripts/evaluate_on_dev.py

# 체크포인트 직접 지정
python scripts/evaluate_on_dev.py --ckt_path checkpoints/260313_run_001/epoch06_0.7498

# beam4/beam8/MBR/TTA 전체 비교 (--run_all)
python scripts/evaluate_on_dev.py --ckt_path <path> --run_all --n_tta_ways 2

# 결과 저장 경로 지정
python scripts/evaluate_on_dev.py --run_all --output_csv prediction/dev_eval_results.csv
```

출력 항목:
- **A. beam4 기본**: 후처리 전/후 ROUGE 비교 및 개선 delta
- **B. beam8** (`--run_all`): 후처리 후 ROUGE
- **C. MBR** (`--run_all`): n_samples=10 샘플링 후 MBR 선택
- **D. TTA** (`--run_all`): n_tta_ways-way 변형본 앙상블
- 최종 비교 테이블 → `prediction/dev_eval_results.csv` 저장

### 5-3. 대회 공식 채점 방식 (다중 정답 ROUGE)

대회는 정답 3개에 대해 각각 ROUGE를 계산한 뒤 평균을 합산합니다.
로컬에서 재현하려면 `evaluate_multi_ref()`를 사용합니다:

```python
from src.utils.metrics import evaluate_multi_ref

scores = evaluate_multi_ref(
    predictions=["김씨가 약속을 잡았다.", ...],
    multi_refs=[
        ["ref1_A", "ref1_B", "ref1_C"],   # 샘플 1의 정답 3개
        ["ref2_A", "ref2_B", "ref2_C"],   # 샘플 2의 정답 3개
        ...
    ],
    use_korouge=True,   # 한국어 보존 ROUGE 권장
)
# scores["rouge_combined"] ≈ 대회 공식 점수 스케일 (47~60)
```

단일 샘플 평가:

```python
from src.utils.metrics import compute_multi_ref_rouge

score = compute_multi_ref_rouge(
    prediction="김씨가 약속을 잡았다.",
    references=["김씨가 약속을 잡았다.", "김씨가 내일 만나기로 했다.", "약속이 잡혔다."],
)
```

### 5-4. rouge vs korouge-score 비교

```python
from src.utils.metrics import compare_rouge_modes

result = compare_rouge_modes(predictions, references)
# {"baseline": {"rouge-1": ..., ...}, "korouge": {"rouge-1": ..., ...}}
```

---

## 6. 앙상블 (Ensemble)

### 6-1. WeightedEnsemble — 다수 모델 예측 앙상블

```python
import pandas as pd
from src.ensemble import WeightedEnsemble

pred_kobart = pd.read_csv("prediction/output_kobart.csv")
pred_kot5   = pd.read_csv("prediction/output_kot5.csv")
pred_solar  = pd.read_csv("prediction/output_solar.csv")

ensemble = WeightedEnsemble()

# 가중치 명시
result = ensemble.predict(
    [pred_solar, pred_kot5, pred_kobart],
    weights=[0.5, 0.3, 0.2],
)

# OOF ROUGE 기반 자동 가중치
result = ensemble.predict(
    [pred_solar, pred_kot5, pred_kobart],
    oof_scores=[0.82, 0.74, 0.71],
)

result.to_csv("prediction/output_ensemble.csv", index=False)
```

- 내부적으로 가중치를 후보 복제 횟수로 변환 후 MBRDecoder로 최선 선택

### 6-2. GroupKFoldTrainer — OOF 학습

topic 그룹 기반 5-fold CV로 OOF 예측을 생성합니다.
각 fold마다 `src/train.py` → `src/inference.py`를 subprocess로 자동 실행합니다.

```python
from src.ensemble import GroupKFoldTrainer

trainer = GroupKFoldTrainer(n_splits=5)
oof_df = trainer.train_oof(
    train_csv="data/train.csv",
    cfg_overrides=["model=kot5", "training=baseline"],
    output_dir="checkpoints/kfold",
)
# oof_df: fname, summary, fold 컬럼
```

- fold별 데이터·체크포인트·예측 저장 위치: `checkpoints/kfold/fold_{N}/`
- `general.checkpoints_root`를 fold별 경로로 자동 override해 경로 불일치 방지

### 6-3. MBRDecoder 단독 사용

```python
from src.ensemble import MBRDecoder

decoder = MBRDecoder()
best = decoder.decode(["요약 후보 1", "요약 후보 2", "요약 후보 3"])
```

---

## 7. 체크포인트 관리

### 7-1. 디렉토리 구조

```
checkpoints/
└── 260313_run_001/              # yymmdd_run_NNN
    ├── epoch04_0.6895/
    ├── epoch05_0.7198/
    └── epoch06_0.7498/          # rouge_combined 상위 3개만 유지
```

### 7-2. 저장 위치 변경

`conf/config.yaml`의 `general.checkpoints_root`로 제어합니다:

```bash
python src/train.py general.checkpoints_root=/path/to/my/checkpoints
```

### 7-3. 특정 체크포인트로 추론

```bash
python src/inference.py inference.ckt_path=checkpoints/260313_run_001/epoch06_0.7498
```

### 7-4. 상위 유지 개수 변경

`src/train.py`의 `BestCheckpointCallback` 생성 시 `top_k`를 수정합니다:

```python
BestCheckpointCallback(output_dir=output_dir, top_k=5)
```

---

## 8. WandB 모니터링

### 8-1. 환경 설정

WandB 관련 설정은 모두 `.env`에서 관리합니다. config.yaml에는 별도 wandb 섹션이 없습니다.

```dotenv
WANDB_ENTITY=my-entity
WANDB_PROJECT=nlp-dialogue-summary
```

run name은 `{model.name}_lr{learning_rate}_ep{num_train_epochs}` 형식으로 자동 생성됩니다.
`model.name` 필드는 각 `conf/model/*.yaml`에 정의되어 있습니다.

### 8-2. 오프라인 모드

```bash
WANDB_MODE=offline python src/train.py

# 이후 동기화
wandb sync wandb/offline-run-*/
```

### 8-3. 메트릭 해석

| 지표 | 해석 |
|------|------|
| `eval/rouge_combined` | 체크포인트 선택 기준 (R1+R2+RL, 최대 3.0) |
| `eval/rouge_1_f1` | 0.32 이상이면 양호 (로컬 단일 정답 기준) |
| `train/loss` | 지속 감소 확인 |

> 로컬 `rouge_combined` 스케일(0~3.0)과 대회 점수(47~60)는 다른 단위입니다.
> 대회 점수 = multi-reference(3개) 기준 R1+R2+RL × 100.

---

## 9. 설정 파일 레퍼런스

### 9-1. `conf/config.yaml` (최상위)

```yaml
general:
  data_path: "data"
  checkpoints_root: "checkpoints"   # 학습 체크포인트 저장 루트

tokenizer:
  encoder_max_len: 512
  decoder_max_len: 100
  bos_token: "<s>"
  eos_token: "</s>"
  special_tokens:                   # 총 15개
    - "#Person1#"  ~ "#Person7#"    # 화자 태그 7개
    - "#PhoneNumber#", "#Address#", "#PassportNumber#"
    - "#DateOfBirth#", "#SSN#", "#CardNumber#", "#CarNumber#", "#Email#"

data:
  use_cleaning: false        # true: clean_text() 적용 (Phase 3+)
  use_length_filter: false   # true: filter_by_length() 적용

metrics:
  use_korouge: false                # true: korouge-score (Phase 3+), false: rouge 라이브러리
```

> **주의**: 기존 11개에서 `#Person4#`~`#Person7#` 4개가 추가되어 총 15개입니다.
> 기존 체크포인트를 재사용할 경우 토크나이저 임베딩 크기가 다를 수 있습니다.

### 9-2. `conf/model/*.yaml`

| 파일 | 모델 | architecture |
|------|------|--------------|
| `kobart.yaml` | `digit82/kobart-summarization` (revision: refs/pr/1) | bart |
| `kobart_v2.yaml` | `gogamza/kobart-base-v2` | bart |
| `kot5.yaml` | `psyche/KoT5-summarization` | t5 |
| `pko_t5.yaml` | `paust/pko-t5-large` | t5 |
| `solar_qlora.yaml` | `upstage/SOLAR-10.7B-Instruct-v1.0` | causal_lm |

### 9-3. `conf/training/*.yaml`

| 파일 | lr | epochs | 특이사항 |
|------|----|--------|---------|
| `baseline.yaml` | 1e-5 | 20 | 기본 실험. `use_all_data: false` |
| `full.yaml` | 3e-5 | 50 | T5 계열. `label_smoothing=0.1` |
| `qlora.yaml` | 2e-5 | 5 | SOLAR. `bf16=true`, `gradient_accum=4` |
| `mps.yaml` | (baseline 상속) | (상속) | Apple Silicon. `batch=2`, `gradient_accum=16` |

공통 주요 키:

```yaml
use_all_data: false          # true: train+dev 합산 학습 (최종 제출 전)
early_stopping_patience: 3
early_stopping_threshold: 0.001
metric_for_best_model: rouge_combined
```

### 9-4. `conf/inference/*.yaml`

| 파일 | 방식 | 주요 설정 |
|------|------|---------|
| `beam4.yaml` | Beam Search | `num_beams=4`, `n_tta_ways=1` |
| `beam8.yaml` | Beam Search | `num_beams=8`, `length_penalty=1.2` |
| `mbr.yaml` | MBR Decoding | `do_sample=true`, `n_samples=10` |
| `tta.yaml` | TTA + MBR | `n_tta_ways=2`, beam4 상속 |
| `solar_api.yaml` | Solar Chat API (Few-shot) | `inference_type: solar_api`, `use_bm25: true` |
| `zero_shot_solar.yaml` | Solar Chat API (Zero-shot) | `prompt_style: zero_shot`, `n_few_shot: 0` |

---

## 10. 모델별 운영 가이드

### 10-1. KoBART (`digit82/kobart-summarization`)

```bash
python src/train.py model=kobart training=baseline
python src/inference.py model=kobart inference.ckt_path=checkpoints/.../epoch##_score
```

### 10-2. KoBART v2 (`gogamza/kobart-base-v2`)

```bash
python src/train.py model=kobart_v2 training=baseline
```

### 10-3. KoT5-summarization

```bash
python src/train.py model=kot5 training=baseline
python src/inference.py model=kot5 inference.ckt_path=checkpoints/.../epoch##_score
```

### 10-4. pko-T5-large

```bash
python src/train.py model=pko_t5 training=full
```

- `conf/model/pko_t5.yaml`에 `prefix: "summarize: "`가 설정되어 인코더 입력에 자동 prepend됩니다.

### 10-5. SOLAR-10.7B (QLoRA)

```bash
# GPU 메모리 24GB 이상 권장
python src/train.py model=solar_qlora training=qlora
```

- 4-bit NF4 양자화 (bitsandbytes)
- LoRA: `r=64`, `alpha=128`, `target_modules: [q_proj, v_proj]`
- `device_map="auto"`로 로드되므로 별도 `.to(device)` 불필요
- **Causal LM 전용 데이터 포맷**: `DatasetForCausalLM`을 사용하며, 각 샘플을 아래 형태의 단일 시퀀스로 구성합니다. prompt 위치의 labels는 `-100`으로 마스킹해 요약문 토큰에 대해서만 loss를 계산합니다.
  ```
  [INST] 다음 대화를 한국어로 요약하세요:\n{dialogue}\n[/INST]\n{summary}</s>
  ```
  `max_length = encoder_max_len(512) + decoder_max_len(100) = 612` 토큰으로 패딩됩니다.
- **표준 `Trainer` 사용**: `Seq2SeqTrainer` 대신 `Trainer + TrainingArguments`를 사용하며, `predict_with_generate`를 사용하지 않습니다. 체크포인트 선택 기준은 ROUGE 대신 `eval_loss` (낮을수록 우수)입니다.

---

## 11. 실험 자동화 스크립트

### 11-1. `scripts/run_all_experiments.sh`

PRD Phase 1~5 전체 실험을 자동으로 순차 실행합니다.

```bash
# 전체 실행 (Phase 1 → 2 → 3 → 5, Phase 4는 API key 필요로 기본 제외)
bash scripts/run_all_experiments.sh

# 특정 Phase만 실행
bash scripts/run_all_experiments.sh phase1
bash scripts/run_all_experiments.sh phase2
bash scripts/run_all_experiments.sh phase3
bash scripts/run_all_experiments.sh phase4   # UPSTAGE_API_KEY 필요
bash scripts/run_all_experiments.sh phase5
```

각 Phase 내용:

| Phase | 내용 | 로그 |
|-------|------|------|
| phase1 | KoBART 베이스라인 학습 + beam4 추론 | `logs/phase1_*.log` |
| phase2 | KoT5, kobart-v2, pko-T5-large 학습 + Hydra sweep | `logs/phase2_*.log` |
| phase3 | EDA 증강, 클리닝+필터 학습, 증강 데이터 학습, Train+Dev 합산 학습 | `logs/phase3_*.log` |
| phase4 | Solar API zero-shot + few-shot 추론 | `logs/phase4_*.log` |
| phase5 | beam4/8/MBR/TTA 비교, test 추론 3종, 앙상블 | `logs/phase5_*.log` |

- 전체 로그: `logs/run_all.log`
- best checkpoint 자동 탐색: `scripts/evaluate_on_dev.find_best_checkpoint()`

---

## 12. 단위 테스트

`tests/test_pipeline.py`에 43개 단위 테스트가 포함되어 있습니다. GPU 없이 실행 가능합니다.

```bash
cd /data/ephemeral/home/NLP
python -m pytest tests/test_pipeline.py -v
```

커버 항목:

| 클래스 | 테스트 대상 |
|--------|------------|
| `TestGetDevice` | CUDA/MPS/CPU 자동 감지 |
| `TestPreprocess` | `make_input()` — train/test 모드, BOS/EOS, prefix |
| `TestCleanText` | 단독 자음, 빈 괄호, 반복 특수기호, 특수 토큰 보존 |
| `TestFilterByLength` | 긴 dialogue/짧은 summary 제거, 인덱스 리셋 |
| `TestPostprocess` | 특수 토큰 제거, 마침표 보장, 중복 문장 제거, 최소 길이 플래그 |
| `TestComputeMetrics` | 완벽 점수, 0점, combined 합산, multi-ref ROUGE |
| `TestDatasets` | `DatasetForSeq2Seq` 길이 및 키 구조 |
| `TestCompareRougeModes` | baseline/korouge 두 모드 반환, float 타입 |
| `TestTTA` | `reverse_utterances()`, `apply_tta()` 2-way/1-way |
| `TestMBRDecoder` | 정상/빈/단일 후보 처리 |
| `TestEvaluateMultiRef` | 단일 정답 일치, 점수 범위 (0~3.0) |

---

## 13. 트러블슈팅

### 13-1. `decoder_start_token_id` 오류 (eval 단계)

```
ValueError: decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation.
```

**원인**: transformers ≥ 4.38에서 `Seq2SeqTrainingArguments(generation_config=...)` 에 커스텀 `GenerationConfig`를 전달하면 모델의 기본 generation config를 완전히 대체합니다. 커스텀 config에 `decoder_start_token_id`가 누락되면 eval 시 오류가 발생합니다.

**해결**: `src/train.py`의 `_build_generation_config()` 함수가 아래 폴백 체인으로 `decoder_start_token_id`를 자동 해석합니다.

```
model.config.decoder_start_token_id
  → tokenizer.bos_token_id
  → tokenizer.pad_token_id
```

모델별 해석 결과:
- KoBART: `model.config.decoder_start_token_id = 2`
- KoT5: `tokenizer.bos_token_id = 0` (fallback)

이 함수는 모델/토크나이저 로드 **이후** 호출되어야 합니다 (`gen_config = _build_generation_config(cfg, tokenizer=tokenizer, model=model)`).

---

### 13-2. `torch.load` 보안 오류

```
ValueError: ... requires PyTorch >= 2.6 ...
```

`conf/model/kobart.yaml`에 `revision: "refs/pr/1"` 지정 → safetensors 포맷 사용.

---

### 13-3. GPU OOM (Out of Memory)

```bash
python src/train.py \
    training.per_device_train_batch_size=2 \
    training.gradient_accumulation_steps=8
```

---

### 13-4. MPS 환경에서 fp16 경고

```
[Train] fp16=True 설정이 무시됩니다 (device=mps). fp32로 학습합니다.
```

정상 동작입니다. `training=mps`를 사용하면 배치를 자동으로 줄여줍니다.

---

### 13-5. WandB 연결 오류

```bash
WANDB_MODE=offline python src/train.py
wandb sync wandb/offline-run-*/
```

---

### 13-6. korouge-score 설치

```bash
pip install korouge-score
```

이후 `conf/config.yaml` 또는 CLI override로 활성화:

```bash
python src/train.py metrics.use_korouge=true
```

---

### 13-7. Causal LM 학습 시 `batch_size mismatch` 오류

```
ValueError: Expected input batch_size (2048) to match target batch_size (400).
```

**원인**: SOLAR 등 decoder-only Causal LM은 `input_ids`와 `labels`가 동일한 shape이어야 합니다. 기존 `DatasetForSeq2Seq`는 encoder `input_ids`(512토큰)와 decoder `labels`(100토큰)를 별도 텐서로 반환하므로 shape mismatch가 발생합니다.

**해결**: `architecture: causal_lm` 설정 시 `_prepare_datasets()`가 자동으로 `DatasetForCausalLM`을 사용하도록 분기합니다. prompt + response를 단일 시퀀스로 합쳐 동일 shape을 보장하며, `Seq2SeqTrainer` 대신 표준 `Trainer`를 사용합니다. 별도 설정 없이 `model=solar_qlora training=qlora`로 실행하면 됩니다.

---

### 13-8. Solar API 추론 분기 오류

`inference=solar_api` 설정에 `inference_type: solar_api` 키가 있어야 합니다.
`conf/inference/solar_api.yaml` 및 `zero_shot_solar.yaml`에 이미 포함되어 있습니다.

---

### 13-9. `evaluate_on_dev.py` 상대 경로 오류

```
HFValidationError: Repo id must be in the form 'repo_name' ...
```

`load_model_tokenizer()`가 상대 경로를 절대 경로로 자동 변환합니다 (`os.path.join(_ROOT, ckt_path)`).
`--ckt_path` 인수에 `checkpoints/260313_run_001/epoch06_0.7498` 형태(프로젝트 루트 기준 상대 경로)로 전달해도 정상 동작합니다.

---

## 빠른 참고 명령어

```bash
# ── 증강 (학습 전 선택) ────────────────────────────────────────
python src/data/run_augment.py --method eda                      # EDA만
python src/data/run_augment.py --method back_translation         # 역번역만
python src/data/run_augment.py --method all                      # 둘 다

# ── 학습 ──────────────────────────────────────────────────────
python src/train.py                                              # KoBART 기본
python src/train.py model=kot5                                   # KoT5
python src/train.py model=pko_t5 training=full                   # pko-T5 풀 학습
python src/train.py training.learning_rate=3e-5                  # LR override
python src/train.py data.use_cleaning=true                       # 클리닝 활성화
python src/train.py data.use_cleaning=true data.use_length_filter=true  # 클리닝+필터
python src/train.py training.use_all_data=true                   # Train+Dev 합산 (최종 제출)
python src/train.py -m model=kobart,kot5                         # 순차 sweep
python src/train.py metrics.use_korouge=true                     # 한국어 ROUGE

# ── 추론 ──────────────────────────────────────────────────────
python src/inference.py inference.ckt_path=checkpoints/PATH      # Beam4
python src/inference.py inference=beam8 inference.ckt_path=...   # Beam8
python src/inference.py inference=mbr   inference.ckt_path=...   # MBR
python src/inference.py inference=tta   inference.ckt_path=...   # TTA (2-way)
python src/inference.py inference=solar_api                      # Solar API (few-shot)
python src/inference.py inference=zero_shot_solar                # Solar API (zero-shot)

# ── Dev 평가 ──────────────────────────────────────────────────
python scripts/evaluate_on_dev.py                                # beam4 기본 평가
python scripts/evaluate_on_dev.py --run_all                      # 전체 비교

# ── 전체 실험 자동화 ───────────────────────────────────────────
bash scripts/run_all_experiments.sh                              # Phase 1~5 전체
bash scripts/run_all_experiments.sh phase3                       # Phase 3만

# ── 단위 테스트 ──────────────────────────────────────────────
python -m pytest tests/test_pipeline.py -v                       # 43개 테스트

# ── 체크포인트 확인 ────────────────────────────────────────────
ls -lt checkpoints/
ls checkpoints/260313_run_001/
```
