---
name: nlp-dialogue-summarization
description: Complete NLP pipeline for Korean dialogue summarization tasks using HuggingFace Transformers + Hydra + WandB. Use this skill whenever the user needs to (1) build or extend a dialogue summarization pipeline (seq2seq, encoder-decoder), (2) fine-tune BART/T5/PEGASUS or other generation models on Korean conversation data, (3) preprocess multi-turn dialogue datasets with special tokens, (4) evaluate summarization quality with ROUGE metrics using morpheme-based tokenization for Korean, (5) run inference and generate submission files for competition, (6) experiment with model selection, hyperparameter tuning, or ensemble strategies for NLP generation tasks, or (7) run hyperparameter sweeps and multi-run experiments with Hydra. Always use this skill when the user mentions 대화 요약, DialogSum, BART fine-tuning, seq2seq training, ROUGE evaluation, KoBART, Hydra sweep, or any Korean NLP summarization competition.
---

# NLP Dialogue Summarization Pipeline

HuggingFace Transformers 기반 한국어 대화 요약 end-to-end 파이프라인.
대회 베이스라인(KoBART + Seq2SeqTrainer + WandB)을 기준으로 **Hydra를 통한 실험 관리**를 지원합니다.

## 프로젝트 구조

```
NLP/
├── conf/                    # Hydra config 디렉토리
│   ├── config.yaml          # 기본 config (defaults 정의)
│   ├── model/               # 모델별 config
│   │   ├── kobart.yaml
│   │   ├── kot5.yaml
│   │   └── pegasus.yaml
│   ├── training/            # 학습 설정
│   │   ├── baseline.yaml
│   │   └── full.yaml
│   └── inference/           # 추론 설정
│       ├── beam4.yaml
│       └── beam8.yaml
├── src/
│   ├── data/
│   │   ├── preprocess.py    # Preprocess 클래스, Dataset 클래스
│   │   └── datamodule.py    # (선택) LightningDataModule 래퍼
│   ├── models/
│   │   └── summarizer.py    # 모델 로드 및 토크나이저 설정
│   ├── train.py             # 학습 메인 스크립트 (@hydra.main)
│   ├── inference.py         # 추론 및 submission 생성
│   └── utils/
│       ├── metrics.py       # ROUGE 계산 (한국어 형태소 기반)
│       └── postprocess.py   # 생성 토큰 제거 등 후처리
├── data/                    # train.csv, dev.csv, test.csv
├── outputs/                 # Hydra 실험 결과 (자동 생성)
├── multirun/                # Sweep 결과 (자동 생성)
├── checkpoints/             # 모델 체크포인트
└── prediction/              # 추론 결과 저장
```

## 파일 위치 규칙
- 새로운 모델: `src/models/` 에 추가
- 새로운 데이터 로더: `src/data/` 에 추가
- 유틸리티 함수: `src/utils/` 에 추가
- **Hydra config**: `conf/` 디렉토리에 config group별로 분리

---

## 핵심 컴포넌트

### 1. Hydra Config 구조

**Hydra를 사용한 계층적 config 관리로 실험 효율성 극대화.**

#### `conf/config.yaml` (메인 설정)
```yaml
defaults:
  - model: kobart              # conf/model/kobart.yaml 선택
  - training: baseline         # conf/training/baseline.yaml 선택
  - inference: beam4           # conf/inference/beam4.yaml 선택
  - _self_                     # 현재 파일 우선순위 최상위

general:
  data_path: "../data/"
  output_dir: "./"
  seed: 42

tokenizer:
  encoder_max_len: 512
  decoder_max_len: 100
  bos_token: "<s>"
  eos_token: "</s>"
  special_tokens:
    - "#Person1#"
    - "#Person2#"
    - "#Person3#"
    - "#PhoneNumber#"
    - "#Address#"
    - "#PassportNumber#"

wandb:
  entity: "your_entity"
  project: "dialogue_summarization"
  name: "${model.name}_${training.learning_rate}"  # 동적 run name

# Hydra 실행 설정
hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
    subdir: ${hydra.job.num}
```

#### `conf/model/kobart.yaml` (모델별 config)
```yaml
name: kobart
model_name: "digit82/kobart-summarization"
architecture: "bart"
```

#### `conf/model/kot5.yaml`
```yaml
name: kot5
model_name: "paust/pko-t5-large"
architecture: "t5"
prefix: "summarize: "  # T5는 prefix 필요
```

#### `conf/training/baseline.yaml` (학습 설정)
```yaml
num_train_epochs: 20
learning_rate: 1e-5
per_device_train_batch_size: 50
per_device_eval_batch_size: 32
warmup_ratio: 0.1
weight_decay: 0.01
lr_scheduler_type: "cosine"
optim: "adamw_torch"
gradient_accumulation_steps: 1
evaluation_strategy: "epoch"
save_strategy: "epoch"
save_total_limit: 5
fp16: true
load_best_model_at_end: true
predict_with_generate: true
generation_max_length: 100
early_stopping_patience: 3
early_stopping_threshold: 0.001
report_to: "wandb"
```

#### `conf/training/full.yaml` (전체 학습)
```yaml
defaults:
  - baseline

num_train_epochs: 50
learning_rate: 3e-5
per_device_train_batch_size: 32
early_stopping_patience: 5
```

#### `conf/inference/beam4.yaml` (추론 설정)
```yaml
num_beams: 4
no_repeat_ngram_size: 2
early_stopping: true
generate_max_length: 100
batch_size: 32
remove_tokens: ["<usr>", "<s>", "</s>", "<pad>"]
ckt_path: "checkpoints/best_model"
result_path: "./prediction/"
```

#### `conf/inference/beam8.yaml`
```yaml
defaults:
  - beam4

num_beams: 8  # beam search 강화
batch_size: 16  # 메모리 고려
```

---

### 2. 데이터 전처리 (`src/data/preprocess.py`)

**데이터 구조:**
- `fname`: 파일명 (train_0, train_1, ...)
- `dialogue`: 멀티턴 대화 (`#Person1#:`, `#Person2#:` 형식)
- `summary`: 요약문 (train/dev 전용)

**Preprocess 클래스 핵심 패턴:**
```python
class Preprocess:
    def make_input(self, dataset, is_test=False):
        # Train/Val: encoder=dialogue, decoder_input=BOS+summary, decoder_output=summary+EOS
        # Test: encoder=dialogue, decoder_input=[BOS] * len
```

**Dataset 클래스 3종:**
- `DatasetForTrain` - (encoder_input, decoder_input, labels)
- `DatasetForVal` - 동일 구조
- `DatasetForInference` - (encoder_input, test_id) — labels 없음

---

### 3. 모델 로드 (`src/models/summarizer.py`)

```python
# Special tokens 추가 필수!
special_tokens_dict = {'additional_special_tokens': config['tokenizer']['special_tokens']}
tokenizer.add_special_tokens(special_tokens_dict)
generate_model.resize_token_embeddings(len(tokenizer))  # 반드시 재구성
```

**지원 모델 목록은** `references/models.md` 참고.

---

### 4. 학습 (`src/train.py`)

**Hydra 데코레이터로 config 관리. HuggingFace `Seq2SeqTrainer` 사용.**

```python
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Hydra가 자동으로 config 로드
    print(OmegaConf.to_yaml(cfg))

    # WandB run name 동적 생성
    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.wandb.name,  # "${model.name}_${training.learning_rate}" 자동 치환
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    # 모델 로드
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    # Trainer 설정
    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
            num_train_epochs=cfg.training.num_train_epochs,
            learning_rate=cfg.training.learning_rate,
            per_device_train_batch_size=cfg.training.per_device_train_batch_size,
            # ... 나머지 설정
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=cfg.training.early_stopping_patience,
            early_stopping_threshold=cfg.training.early_stopping_threshold
        )]
    )

    trainer.train()

if __name__ == "__main__":
    main()
```

**핵심 패턴:**
- `@hydra.main` 데코레이터로 config 자동 로드
- `cfg.model.name`, `cfg.training.learning_rate` 등으로 접근
- Hydra가 자동으로 실험 결과를 `outputs/` 디렉토리에 저장
- Command-line override 지원 (`python train.py training.learning_rate=5e-5`)

---

### 5. ROUGE 평가 (`src/utils/metrics.py`)

**한국어 특성 주의:** 형태소 단위 토크나이징으로 점수 산출.

```python
from rouge import Rouge

def compute_metrics(config, tokenizer, pred):
    rouge = Rouge()
    # 불필요한 생성 토큰 제거 후 평가
    results = rouge.get_scores(predictions, labels, avg=True)
    return {key: value["f"] for key, value in results.items()}
```

**대회 평가 지표:**
```
Score = mean(ROUGE-1-F1) + mean(ROUGE-2-F1) + mean(ROUGE-L-F1)
```
test 데이터는 dialogue 1개당 summary 3개 → 3개 개별 채점 후 종합.

---

### 6. 추론 (`src/inference.py`)

```python
# Beam search로 요약문 생성
output = generate_model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
    early_stopping=config['inference']['early_stopping'],
    max_length=config['inference']['generate_max_length'],
    num_beams=config['inference']['num_beams'],
)
```

---

## 실험 전략

자세한 모델별 전략은 `references/models.md` 참고.

| 전략 | 설명 |
|------|------|
| 모델 교체 | KoBART → PEGASUS-ko, mBART, KoT5 등 |
| 프롬프트 엔지니어링 | 디코더 입력 형식 변경 |
| 후처리 | 특수 토큰 정리, 문장 정규화 |
| 앙상블 | 여러 체크포인트 출력 평균화 |
| 데이터 증강 | back-translation, paraphrase |

---

## WandB 설정

```bash
# .env 파일에 추가 (git 커밋 금지!)
WANDB_API_KEY=your-api-key
WANDB_PROJECT=dialogue_summarization
WANDB_ENTITY=your-username
```

Run name은 Hydra interpolation으로 자동 생성: `"${model.name}_${training.learning_rate}"`

---

## 빠른 시작

### 초기 설정

```bash
# 1. 환경 변수 설정
cp references/.env.template .env
# .env 파일을 열어서 WandB API key와 entity/project 설정

# 2. 의존성 설치
pip install -r references/requirements.txt
# 또는 최소 설치:
# pip install torch transformers hydra-core omegaconf wandb rouge pandas python-dotenv

# 3. Git 설정 (프로젝트 루트에서)
cp references/.gitignore .gitignore

# 4. Hydra config 디렉토리 생성
mkdir -p conf/model conf/training conf/inference
```

### 기본 실행
```bash
# 1. 환경 변수 로드 확인
cat .env

# 2. Config 수정
# conf/config.yaml의 data_path, wandb entity/project 설정

# 3. 기본 학습 (kobart + baseline)
python src/train.py

# 4. 모델 변경
python src/train.py model=kot5

# 5. 학습 설정 변경
python src/train.py training=full

# 6. Command-line override
python src/train.py training.learning_rate=5e-5 training.num_train_epochs=30

# 7. 여러 설정 동시 변경
python src/train.py model=pegasus training=full inference=beam8

# 8. 추론
python src/inference.py inference.ckt_path=outputs/2024-03-10/12-34-56/checkpoints/best_model
```

### Hydra Sweep (하이퍼파라미터 탐색)

```bash
# 1. Learning rate sweep
python src/train.py -m training.learning_rate=1e-5,3e-5,5e-5

# 2. 여러 모델 비교
python src/train.py -m model=kobart,kot5,pegasus

# 3. Grid search (learning rate × batch size)
python src/train.py -m \
  training.learning_rate=1e-5,3e-5,5e-5 \
  training.per_device_train_batch_size=32,50

# 4. 모델별 최적 설정 실험
python src/train.py -m \
  model=kobart,kot5 \
  training=baseline,full

# 결과는 multirun/YYYY-MM-DD/HH-MM-SS/0, 1, 2, ... 에 저장됨
```

### Hydra Override 패턴

```bash
# Config group 변경
python train.py model=kot5                    # conf/model/kot5.yaml 사용

# 특정 값만 override
python train.py training.learning_rate=5e-5   # 해당 값만 변경

# 리스트 항목 추가
python train.py +tokenizer.special_tokens=["#NewToken#"]

# 값 제거
python train.py ~wandb                        # wandb 설정 비활성화

# Nested 값 변경
python train.py hydra.run.dir=my_output       # Hydra 출력 디렉토리 변경
```

---

## Hydra Sweep 고급 패턴

### Optuna Integration (자동 하이퍼파라미터 최적화)

```yaml
# conf/config.yaml에 추가
defaults:
  - override hydra/sweeper: optuna

hydra:
  sweeper:
    optuna_config:
      study_name: dialogue_summarization
      direction: maximize  # ROUGE 점수 최대화
      n_trials: 20
      n_jobs: 1
    params:
      training.learning_rate: range(1e-6, 1e-4, log=True)
      training.warmup_ratio: choice(0.05, 0.1, 0.15)
      training.weight_decay: range(0.0, 0.1)
```

```bash
# Optuna sweep 실행
pip install hydra-optuna-sweeper
python src/train.py -m
```

### Joblib Parallel Launcher (병렬 실행)

```yaml
# conf/config.yaml에 추가
defaults:
  - override hydra/launcher: joblib

hydra:
  launcher:
    n_jobs: 4  # 4개 병렬 실행
```

```bash
pip install hydra-joblib-launcher
python src/train.py -m model=kobart,kot5,pegasus training.learning_rate=1e-5,3e-5
# 총 6개 실험이 4개씩 병렬로 실행됨
```

---

## 참고 문서

### 가이드
- `references/setup_guide.md` — **[시작]** 프로젝트 초기 설정 가이드 (환경 설정, 패키지 설치, 문제 해결)
- `references/models.md` — 모델 선택 가이드 및 HuggingFace 모델 목록
- `references/training_tips.md` — 학습 최적화, 디버깅 팁, Hydra best practices, 자주 발생하는 오류
- `references/hydra_sweep.md` — Hydra sweep 전략, Optuna 통합, 병렬 실행, 결과 분석 가이드

### 템플릿 파일
- `references/.env.template` — 환경 변수 템플릿 (WandB 설정 등)
- `references/requirements.txt` — 필수 패키지 목록 (Hydra, Transformers, WandB 등)
- `references/.gitignore` — Git 제외 파일 목록 (.env, checkpoints, outputs 등)

**사용법:**
```bash
cp references/.env.template .env          # 환경 변수 설정
pip install -r references/requirements.txt  # 패키지 설치
cp references/.gitignore .gitignore       # Git 설정
```

---

## 핵심 장점

✅ **Hydra로 실험 효율성 극대화**
- Config group으로 모델/학습/추론 설정 모듈화
- Command-line override로 빠른 실험
- Sweep으로 자동 하이퍼파라미터 탐색
- 실험 결과 자동 저장 및 재현 가능

✅ **Seq2SeqTrainer로 안정적 학습**
- HuggingFace 생태계 완벽 호환
- Generate 기능 내장
- ROUGE 평가 자동화

✅ **WandB로 실험 추적**
- 모든 metric 자동 로깅
- Sweep 결과 시각화
- 팀 협업 지원
