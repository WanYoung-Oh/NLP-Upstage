# 한국어 대화 요약 (Dialogue Summarization)

> **목표**: ROUGE-F1 베이스라인 47.12 → **60+**
> Upstage AI Stages 대회 제출용 한국어 대화 요약 파이프라인

---

## 프로젝트 구조

```
NLP/
├── conf/                        # Hydra 설정 파일
│   ├── config.yaml              # 최상위 기본 설정
│   ├── model/                   # 모델별 설정
│   │   ├── kobart.yaml          # KoBART (digit82/kobart-summarization)
│   │   ├── kobart_v2.yaml       # KoBART v2 (gogamza/kobart-base-v2)
│   │   ├── kot5.yaml            # KoT5-summarization
│   │   ├── pko_t5.yaml          # pko-T5-large
│   │   └── solar_qlora.yaml     # SOLAR-KO-10.7B (QLoRA)
│   ├── training/                # 학습 하이퍼파라미터 프리셋
│   │   ├── baseline.yaml        # 기본 설정 (lr=1e-5, epoch=20)
│   │   ├── full.yaml            # 풀 학습 (lr=3e-5, epoch=50)
│   │   └── qlora.yaml           # QLoRA 학습 설정
│   └── inference/               # 추론 설정
│       ├── beam4.yaml           # Beam Search (beam=4)
│       ├── beam8.yaml           # Beam Search (beam=8)
│       ├── mbr.yaml             # MBR Decoding
│       └── solar_api.yaml       # Solar Chat API
├── src/
│   ├── train.py                 # 학습 진입점
│   ├── inference.py             # 추론 진입점
│   ├── ensemble.py              # 앙상블
│   ├── data/
│   │   ├── preprocess.py        # 전처리 (토크나이징, 데이터셋 클래스)
│   │   └── augment.py           # 데이터 증강 (EDA/AEDA)
│   ├── models/
│   │   └── summarizer.py        # 모델/토크나이저 로더
│   └── utils/
│       ├── metrics.py           # ROUGE 평가 (rouge / korouge-score)
│       ├── postprocess.py       # 후처리 파이프라인
│       └── device.py            # GPU/CPU 디바이스 감지
├── data/
│   ├── train.csv                # 학습 데이터 (12,457건)
│   ├── dev.csv                  # 검증 데이터 (499건, summary 1개)
│   └── test.csv                 # 테스트 데이터 (제출용, summary 없음)
├── checkpoints/                 # 학습 체크포인트 (yymmdd_run_NNN/)
├── prediction/                  # 추론 결과 CSV
├── outputs/                     # Hydra 실행 로그 (.hydra/ + train.log)
├── wandb/                       # WandB 로컬 로그
├── docs/                        # 프로젝트 문서
└── requirements.txt
```

---

## 구현 완료 항목

### 핵심 파이프라인
- [x] **학습 파이프라인** (`src/train.py`) — Hydra + WandB + Seq2SeqTrainer
- [x] **추론 파이프라인** (`src/inference.py`) — Beam Search / MBR Decoding / Solar API
- [x] **전처리** (`src/data/preprocess.py`) — 토크나이저 래핑, 특수 토큰 처리, TTA
- [x] **후처리** (`src/utils/postprocess.py`) — 특수 토큰 제거, 반복 문장 제거

### 모델 지원
| 모델 | 구분 | 상태 |
|------|------|------|
| `digit82/kobart-summarization` | Seq2Seq | ✅ 학습 완료 (safetensors 사용) |
| `gogamza/kobart-base-v2` | Seq2Seq | ⬜ 미학습 |
| `EbanLee/kobart-summary-v3` | Seq2Seq | ⬜ 미학습 |
| `paust/pko-t5-large` | Seq2Seq | ⬜ 미학습 |
| `SOLAR-KO-10.7B` | CausalLM (QLoRA) | ⬜ 미학습 |
| Solar Chat API | API | ⬜ 미검증 |

### 평가 메트릭
- [x] ROUGE-1 / ROUGE-2 / ROUGE-L F1 개별 기록
- [x] `rouge_combined` = R1 + R2 + RL 합산 점수 (체크포인트 선택 기준)
- [x] WandB `eval/` 그룹 하단에 epoch 기준 x축으로 표시
- [x] `korouge-score` (한국어 문자 보존) 전환 플래그 지원

### 체크포인트 관리
- [x] `checkpoints/{yymmdd_run_NNN}/` 자동 run ID 부여
- [x] `epoch{##}_{score:.4f}` 형식 디렉토리명
- [x] 상위 3개(best) 체크포인트만 유지 (`BestCheckpointCallback`)
- [x] `rouge_combined` 기준 early stopping

### 인프라
- [x] Hydra config 계층 (`model` / `training` / `inference` 분리)
- [x] `outputs/` 디렉토리 — Hydra 전용 (`.hydra/` + `train.log`)
- [x] WandB `dir=_PROJECT_ROOT` 고정 (Hydra CWD 변경 무관)
- [x] `safetensors` 자동 fallback (`refs/pr/1` revision)

---

## 남은 검증 작업

### 베이스라인 검증
- [x] `python src/train.py` — 오류 없이 학습 시작
- [x] WandB 대시보드 run 생성 및 ROUGE 메트릭 표시
- [ ] **Dev ROUGE-1 목표 47.12 달성** — 현재 최고 32.54% (학습 진행 중)
- [ ] `python src/inference.py` — `prediction/output.csv` 정상 생성 검증
- [x] `outputs/` 디렉토리 Hydra 전용 확인

### Phase 2 — 모델 실험
- [ ] KoT5-summarization 학습 & Dev ROUGE 기록
- [ ] kobart-base-v2 학습 & Dev ROUGE 기록
- [ ] pko-T5-large 학습 & Dev ROUGE 기록
- [ ] 모델별 ROUGE 비교 후 최고 모델 선정

### Phase 3 — 품질 개선
- [ ] `USE_KOROUGE = True` 전환 후 평가 모드 비교
- [ ] 텍스트 클리닝 (`clean_text`) 효과 측정
- [ ] 데이터 증강 (`augment.py`) 적용 및 ROUGE 변화 측정

### Phase 4 — 고급 기법
- [ ] SOLAR QLoRA 학습 & 추론
- [ ] Solar Chat API few-shot 추론
- [ ] MBR Decoding vs Beam Search 비교
- [ ] 앙상블 (`ensemble.py`) 효과 측정

---

## 빠른 시작

### 환경 설정

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 환경 변수 설정
cp .env.example .env
# .env에 WANDB_ENTITY, WANDB_PROJECT, UPSTAGE_API_KEY 입력
```

### 학습

```bash
# 기본 실행 (KoBART + baseline 설정)
python src/train.py

# 모델 변경
python src/train.py model=kot5

# 하이퍼파라미터 override
python src/train.py training.learning_rate=3e-5 training.num_train_epochs=30
```

### 추론

```bash
# Beam Search 추론
python src/inference.py inference.ckt_path=checkpoints/260313_run_001/epoch10_0.7678

# MBR Decoding
python src/inference.py inference=mbr inference.ckt_path=checkpoints/...
```

---

## 평가 구조

### 로컬 평가 (dev.csv)
| 지표 | WandB 키 | 설명 |
|------|---------|------|
| ROUGE-1 F1 | `eval/rouge_1_f1` | 유니그램 F1 |
| ROUGE-2 F1 | `eval/rouge_2_f1` | 바이그램 F1 |
| ROUGE-L F1 | `eval/rouge_l_f1` | LCS 기반 F1 |
| 합산 | `eval/rouge_combined` | R1+R2+RL (체크포인트 선택 기준) |

### 대회 평가 vs 로컬 평가
- **로컬**: dev.csv 기준 summary **1개** 대비 ROUGE
- **대회**: 평가 데이터 기준 summary **3개** 중 최적 대비 ROUGE → 로컬보다 높게 산출됨
- 목표 ROUGE-1 47.12는 대회 multi-reference 기준 수치

---

## 환경

| 항목 | 버전 |
|------|------|
| Python | 3.10 |
| PyTorch | 2.5.1+cu124 |
| Transformers | 5.3.0 |
| GPU | NVIDIA RTX 3090 |
