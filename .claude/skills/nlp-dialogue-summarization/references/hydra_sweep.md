# Hydra Sweep 전략 가이드

대화 요약 태스크를 위한 효율적인 하이퍼파라미터 탐색 전략.

---

## 기본 Sweep 패턴

### 1. Learning Rate Sweep
```bash
# 3개 learning rate 비교
python src/train.py -m training.learning_rate=1e-5,3e-5,5e-5

# 결과: multirun/YYYY-MM-DD/HH-MM-SS/{0,1,2}/
```

### 2. 모델 비교
```bash
# 3개 모델 동시 학습
python src/train.py -m model=kobart,kot5,pegasus

# WandB에 각각 다른 run name으로 자동 로깅
```

### 3. Grid Search
```bash
# Learning rate × Batch size (3×2 = 6개 실험)
python src/train.py -m \
  training.learning_rate=1e-5,3e-5,5e-5 \
  training.per_device_train_batch_size=32,50
```

### 4. Config Group 조합
```bash
# 모델 × 학습 설정 (2×2 = 4개 실험)
python src/train.py -m \
  model=kobart,kot5 \
  training=baseline,full
```

---

## 대회 맞춤 Sweep 전략

### Stage 1: 빠른 모델 선택 (1-2시간)
```bash
# 적은 epoch로 여러 모델 비교
python src/train.py -m \
  model=kobart,kot5,pegasus \
  training.num_train_epochs=5 \
  training.per_device_train_batch_size=32
```
→ ROUGE 가장 높은 모델 1개 선택

### Stage 2: Learning Rate 탐색 (2-3시간)
```bash
# 선택된 모델로 learning rate sweep
python src/train.py -m \
  model=kobart \
  training.learning_rate=5e-6,1e-5,3e-5,5e-5,1e-4
```
→ Validation ROUGE 최고 learning rate 선택

### Stage 3: 세부 튜닝 (3-5시간)
```bash
# 최적 learning rate로 다른 하이퍼파라미터 탐색
python src/train.py -m \
  model=kobart \
  training.learning_rate=3e-5 \
  training.warmup_ratio=0.05,0.1,0.15 \
  training.weight_decay=0.0,0.01,0.05
```

### Stage 4: Full Training (10-20시간)
```bash
# 최적 설정으로 전체 epoch 학습
python src/train.py \
  model=kobart \
  training.learning_rate=3e-5 \
  training.warmup_ratio=0.1 \
  training.weight_decay=0.01 \
  training.num_train_epochs=50
```

---

## Optuna 기반 자동 최적화

### 설치
```bash
pip install hydra-optuna-sweeper
```

### Config 설정
```yaml
# conf/config.yaml에 추가
defaults:
  - override hydra/sweeper: optuna

hydra:
  sweeper:
    optuna_config:
      study_name: dialogue_summarization_study
      storage: null  # 또는 "sqlite:///optuna.db" (영구 저장)
      direction: maximize  # ROUGE 점수 최대화
      n_trials: 30
      n_jobs: 1
    params:
      # Learning rate (log scale)
      training.learning_rate:
        type: float
        low: 1e-6
        high: 1e-4
        log: true

      # Warmup ratio
      training.warmup_ratio:
        type: categorical
        choices: [0.05, 0.1, 0.15, 0.2]

      # Weight decay
      training.weight_decay:
        type: float
        low: 0.0
        high: 0.1

      # Batch size (power of 2)
      training.per_device_train_batch_size:
        type: categorical
        choices: [16, 32, 50, 64]

      # Scheduler type
      training.lr_scheduler_type:
        type: categorical
        choices: [linear, cosine, polynomial]
```

### 실행
```bash
python src/train.py -m

# 결과 확인
# multirun/YYYY-MM-DD/HH-MM-SS/optimization_results.yaml
```

### 학습 스크립트 수정 (Optuna 연동)
```python
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # ... 학습 코드 ...

    # Validation ROUGE 점수 계산
    metrics = trainer.evaluate()
    rouge_score = metrics['eval_rouge-1-f'] + metrics['eval_rouge-2-f'] + metrics['eval_rouge-l-f']

    # Optuna에 결과 반환 (Hydra가 자동으로 처리)
    return rouge_score  # maximize 방향으로 최적화
```

---

## 병렬 실행 (Joblib Launcher)

### 설치
```bash
pip install hydra-joblib-launcher
```

### Config 설정
```yaml
# conf/config.yaml에 추가
defaults:
  - override hydra/launcher: joblib

hydra:
  launcher:
    n_jobs: 4  # 4개 GPU 또는 CPU 코어 사용
    batch_size: auto
    verbose: 10
```

### 실행
```bash
# 6개 실험을 4개씩 병렬로 실행
python src/train.py -m \
  model=kobart,kot5,pegasus \
  training.learning_rate=1e-5,3e-5

# GPU 메모리 부족 시 CUDA_VISIBLE_DEVICES로 분산
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train.py -m ...
```

---

## Sweep 결과 분석

### WandB에서 비교
```python
import wandb

api = wandb.Api()
runs = api.runs("your_entity/dialogue_summarization")

# 최고 성능 run 찾기
best_run = max(runs, key=lambda r: r.summary.get('eval/rouge-l-f', 0))
print(f"Best run: {best_run.name}")
print(f"Config: {best_run.config}")
print(f"ROUGE-L: {best_run.summary['eval/rouge-l-f']}")
```

### 로컬에서 분석
```bash
# 각 실험의 최종 metric 수집
for dir in multirun/YYYY-MM-DD/HH-MM-SS/*/; do
  echo "$dir:"
  tail -5 "$dir/train.log" | grep "rouge"
done

# 최고 점수 찾기
find multirun/YYYY-MM-DD/HH-MM-SS -name "*.log" \
  | xargs grep "rouge-l" \
  | sort -t':' -k3 -rn \
  | head -1
```

### Hydra 내장 분석 도구
```python
# analysis.py
from hydra import compose, initialize
from omegaconf import OmegaConf

initialize(config_path="../conf")

# 각 실험의 config 로드
sweep_dir = "multirun/2024-03-10/12-34-56"
for i in range(6):
    cfg = OmegaConf.load(f"{sweep_dir}/{i}/.hydra/config.yaml")
    print(f"Run {i}: lr={cfg.training.learning_rate}, model={cfg.model.name}")
```

---

## Sweep 최적화 팁

### 1. Early Stopping 활용
```yaml
# 성능 좋지 않은 실험 조기 종료로 시간 절약
training:
  early_stopping_patience: 3
  early_stopping_threshold: 0.001
```

### 2. Checkpoint 관리
```yaml
# Sweep 시 디스크 용량 절약
training:
  save_strategy: "epoch"
  save_total_limit: 2  # 최근 2개만 유지
  load_best_model_at_end: true
```

### 3. 빠른 실험을 위한 데이터 서브셋
```python
# train.py에서 debug mode 추가
if cfg.get('debug', False):
    train_dataset = train_dataset.select(range(1000))
    val_dataset = val_dataset.select(range(100))
```

```bash
# Quick validation sweep
python src/train.py -m \
  debug=true \
  training.num_train_epochs=2 \
  model=kobart,kot5,pegasus
```

### 4. WandB Sweep 통합 (대안)
```yaml
# wandb_sweep.yaml
program: src/train.py
method: bayes  # random, grid, bayes
metric:
  name: eval/rouge-l-f
  goal: maximize
parameters:
  training.learning_rate:
    distribution: log_uniform_values
    min: 1e-6
    max: 1e-4
  training.warmup_ratio:
    values: [0.05, 0.1, 0.15]
  training.weight_decay:
    distribution: uniform
    min: 0.0
    max: 0.1
```

```bash
wandb sweep wandb_sweep.yaml
wandb agent your_entity/dialogue_summarization/sweep_id
```

---

## 권장 Sweep 순서 요약

1. **빠른 모델 스크리닝** (1-2h): 5 epoch, 3-4개 모델
2. **Learning rate 탐색** (2-3h): 최적 모델, 5-7개 lr
3. **세부 튜닝** (3-5h): warmup, weight_decay, batch_size
4. **Optuna 자동 최적화** (선택, 5-10h): 30-50 trials
5. **Final training** (10-20h): 최적 config, full epochs
6. **앙상블** (선택, 1-2h): Top 3 모델 예측 병합
