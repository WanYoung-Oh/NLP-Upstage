# 학습 최적화 & 트러블슈팅

---

## Hydra 사용 Best Practices

### Config 구조화 원칙
```
conf/
├── config.yaml          # 메인 config, defaults 정의
├── model/               # 모델별 config (kobart, kot5, pegasus)
├── training/            # 학습 설정 (baseline, full, quick)
├── inference/           # 추론 설정 (beam4, beam8, greedy)
└── experiment/          # 전체 실험 조합 (optional)
```

### 실험 재현성 보장
```yaml
# conf/config.yaml
general:
  seed: 42  # 고정된 seed

# Hydra 실행 시 자동으로 .hydra/ 폴더에 사용된 config 저장됨
hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  job:
    chdir: false  # 작업 디렉토리 변경 방지
```

### Command-line Override 우선순위
1. Command-line 인자 (`python train.py training.lr=5e-5`)
2. Config group 선택 (`python train.py model=kot5`)
3. `config.yaml`의 `defaults`
4. 각 config group 파일의 기본값

### WandB와 Hydra 통합 팁
```python
import wandb
from omegaconf import OmegaConf

# Config를 WandB에 dict로 전달
wandb.init(
    config=OmegaConf.to_container(cfg, resolve=True)  # resolve=True로 변수 치환
)

# Run name에 Hydra 실험 번호 포함
wandb.init(
    name=f"{cfg.model.name}_{cfg.training.learning_rate}_{hydra.job.num}"
)
```

### Sweep 결과 분석
```bash
# 모든 실험 결과의 metric 수집
find multirun/YYYY-MM-DD/HH-MM-SS -name "*.log" | xargs grep "rouge"

# 최고 성능 모델 찾기
grep -r "rouge-l" multirun/YYYY-MM-DD/HH-MM-SS/*/metrics.txt | sort -t':' -k2 -rn | head -1
```

---

## 자주 발생하는 오류

### 1. `resize_token_embeddings` 누락
```
RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long; but got torch.FloatTensor
```
→ special token 추가 후 반드시 호출:
```python
generate_model.resize_token_embeddings(len(tokenizer))
```

### 2. ROUGE 계산 오류 (한국어)
```
RecursionError: maximum recursion depth exceeded
```
→ 빈 문자열 예측 방지:
```python
replaced_predictions = [p if p.strip() else "없음" for p in replaced_predictions]
```

### 3. OOM (Out of Memory)
```
RuntimeError: CUDA out of memory
```
→ 대응 방법 (순서대로 시도):
1. `per_device_train_batch_size` 줄이기
2. `gradient_accumulation_steps` 늘리기 (effective batch size 유지)
3. `fp16: true` 설정 확인
4. `encoder_max_len` 줄이기 (512 → 256)

### 4. WandB 연결 오류
```
wandb: ERROR Run failed
```
→ `.env` 파일에 API key 설정 또는:
```bash
wandb login
```
→ 오프라인 모드:
```bash
WANDB_MODE=offline python src/train.py
```

### 5. Hydra Config 오류
```
omegaconf.errors.ConfigAttributeError: Key 'model.name' not found
```
→ Config group 구조 확인:
- `conf/model/kobart.yaml`에 `name: kobart` 필드 존재하는지 확인
- `config.yaml`의 `defaults`에 `- model: kobart` 설정되어 있는지 확인

```
hydra.errors.MissingConfigException: Cannot find primary config 'config'
```
→ `@hydra.main` 데코레이터의 `config_path` 확인:
```python
@hydra.main(version_base=None, config_path="../conf", config_name="config")
# config_path는 train.py 기준 상대 경로
```

### 6. Hydra Working Directory 변경 문제
```
FileNotFoundError: [Errno 2] No such file or directory: '../data/train.csv'
```
→ Hydra는 기본적으로 `outputs/` 디렉토리로 작업 디렉토리 변경
→ 해결 방법:
```yaml
# conf/config.yaml
hydra:
  job:
    chdir: false  # 작업 디렉토리 변경 비활성화
```
또는 절대 경로 사용:
```yaml
general:
  data_path: ${hydra:runtime.cwd}/data/  # 실행 시점의 디렉토리
```

---

## 성능 최적화 팁

### 데이터 전처리
- `max_length` 분포 확인 후 설정 (EDA에서 확인한 right-skewed 분포 참고)
- `encoder_max_len=512`, `decoder_max_len=100`이 베이스라인 기준 (대화 길이 분포상 적절)
- 개인정보 마스킹 토큰(`#Person1#` 등)을 special token으로 등록해야 분해 방지

### 학습 안정성
```yaml
training:
  seed: 42                    # 재현성 보장
  fp16: true                  # Mixed precision (속도↑, 메모리↓)
  load_best_model_at_end: true  # 최고 성능 체크포인트 자동 로드
  early_stopping_patience: 3    # 3 epoch 개선 없으면 중단
```

### 추론 품질
```yaml
inference:
  num_beams: 4               # Greedy(1) vs Beam search(4+)
  no_repeat_ngram_size: 2    # 2-gram 반복 방지
  early_stopping: true       # EOS 토큰 도달 시 중단
  generate_max_length: 100   # 대화 길이의 20% 이내 권장
```

---

## 평가 지표 이해

### ROUGE-F1 공식
```
ROUGE-1-F1: 단어(1-gram) 단위 겹침
ROUGE-2-F1: 2-gram 단위 겹침  
ROUGE-L-F1: 최장 공통 부분 수열(LCS) 기반
```

### 한국어 형태소 토크나이징
- 조사가 띄어쓰기로 분리되지 않는 한국어 특성 고려
- `rouge` 라이브러리는 공백 기준 분리 → 형태소 분리 후 평가 권장
- konlpy나 mecab으로 사전 토크나이징 후 ROUGE 계산 가능:

```python
from konlpy.tag import Mecab
mecab = Mecab()

def morpheme_tokenize(text: str) -> str:
    return " ".join(mecab.morphs(text))

# ROUGE 계산 전 적용
pred_morphemes = morpheme_tokenize(prediction)
gold_morphemes = morpheme_tokenize(gold)
```

### test 데이터 평가 방식
- test 1개 dialogue → summary 3개 (gold)
- 예측 1개 vs gold 3개 → 개별 점수 산출 후 평균

---

## 실험 재현 체크리스트

### Hydra 설정
- [ ] `conf/config.yaml`의 `defaults` 섹션 확인
- [ ] Config group 디렉토리 구조 (`conf/model/`, `conf/training/`, `conf/inference/`) 존재 확인
- [ ] `hydra.job.chdir: false` 설정으로 작업 디렉토리 변경 방지
- [ ] `general.seed` 고정값 설정 (재현성 보장)

### 학습 환경
- [ ] `pip install hydra-core omegaconf` 완료
- [ ] `conf/config.yaml` WandB entity/project 설정
- [ ] `general.data_path` 올바른 경로 확인 (절대 경로 또는 `${hydra:runtime.cwd}` 사용)
- [ ] special tokens 추가 및 `resize_token_embeddings` 호출
- [ ] `.env` 파일 git 커밋 방지 (.gitignore 확인)

### 추론
- [ ] 체크포인트 경로 (`inference.ckt_path`) 설정
  - Hydra 사용 시: `outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/best_model`
- [ ] `remove_tokens` 목록에 해당 모델 토크나이저의 특수 토큰 포함

### Sweep 실험
- [ ] Sweep 실행 시 `-m` 플래그 사용
- [ ] 결과 저장 경로 확인: `multirun/YYYY-MM-DD/HH-MM-SS/`
- [ ] 각 실험의 `.hydra/` 폴더에 사용된 config 자동 저장 확인
