# PRD — 일상 대화 요약 경진대회 구현 계획

> **기반**: `/nlp-dialogue-summarization` 스킬 구조
> **목표**: ROUGE-F1 베이스라인 47.12 → 60+
> **작성일**: 2026-03-12
> **참고**: `docs/RESEARCH.md`

---

## 목차

1. [프로젝트 구조](#1-프로젝트-구조)
2. [Hydra Config 설계](#2-hydra-config-설계)
3. [Phase별 구현 계획](#3-phase별-구현-계획)
   - Phase 1: 환경 셋업 & 베이스라인 재현
   - Phase 2: 모델 업그레이드
   - Phase 3: 데이터/학습 전략 고도화
   - Phase 4: LLM(Solar API) 활용
   - Phase 5: 앙상블 & 후처리
4. [전체 체크리스트](#4-전체-체크리스트)
5. [테스트 계획](#5-테스트-계획)
6. [완료 기준](#6-완료-기준)

---

## 1. 프로젝트 구조

스킬 정의를 기준으로, 아래 구조를 목표로 한다.

```
NLP/
├── conf/
│   ├── config.yaml                  # 메인 config (defaults 정의)
│   ├── model/
│   │   ├── kobart.yaml              # 베이스라인 (digit82/kobart-summarization)
│   │   ├── kobart_v2.yaml           # gogamza/kobart-base-v2
│   │   ├── kot5.yaml                # psyche/KoT5-summarization
│   │   ├── pko_t5.yaml              # paust/pko-t5-large
│   │   └── solar_qlora.yaml         # upstage/SOLAR-10.7B (QLoRA)
│   ├── training/
│   │   ├── baseline.yaml            # 베이스라인 학습 설정
│   │   ├── full.yaml                # 장기 학습 설정
│   │   └── qlora.yaml               # SOLAR QLoRA 전용 설정
│   └── inference/
│       ├── beam4.yaml               # 기본 beam search
│       ├── beam8.yaml               # 강화된 beam search
│       └── mbr.yaml                 # MBR Decoding
├── src/
│   ├── data/
│   │   ├── preprocess.py            # Preprocess 클래스 + Dataset 3종 + 클리닝
│   │   └── augment.py               # 데이터 증강 (back-translation, EDA/AEDA)
│   ├── models/
│   │   └── summarizer.py            # 모델 로드 (BART/T5/CausalLM 분기)
│   ├── train.py                     # @hydra.main, Seq2SeqTrainer
│   ├── inference.py                 # beam search / MBR decoding / Solar API
│   └── utils/
│       ├── device.py                # 디바이스 자동 감지 (NVIDIA GPU / Mac M4 MPS)
│       ├── metrics.py               # 형태소 기반 ROUGE (konlpy Okt)
│       └── postprocess.py           # 특수 토큰 제거, 마침표 보장, 반복 제거
├── data/                            # train.csv, dev.csv, test.csv
├── docs/
│   ├── RESEARCH.md
│   ├── STRATEGY.md
│   └── PRD.md                       # 본 문서
├── outputs/                         # Hydra 실험 결과 (자동 생성)
├── multirun/                        # Sweep 결과 (자동 생성)
├── checkpoints/                     # 모델 체크포인트
├── prediction/                      # 추론 결과 저장
├── .env                             # API key, 경로 설정
└── requirements.txt
```

### 스킬과의 통일성 원칙
- 모든 실험은 `@hydra.main` + `Seq2SeqTrainer` 기반
- 모델별 분기는 `conf/model/` config로 제어 (`architecture: bart | t5 | causal_lm`)
- Solar API inference는 `src/inference.py` 내 별도 함수로 통합
- ROUGE 평가는 `src/utils/metrics.py`의 `compute_metrics`를 단일 진입점으로 통일

---

## 2. Hydra Config 설계

### `conf/config.yaml` — 메인 설정

```yaml
defaults:
  - model: kobart
  - training: baseline
  - inference: beam4
  - _self_

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
    # 베이스라인 토큰
    - "#Person1#"
    - "#Person2#"
    - "#Person3#"
    - "#PhoneNumber#"
    - "#Address#"
    - "#PassportNumber#"
    # 추가 마스킹 토큰 (베이스라인 누락분)
    - "#DateOfBirth#"
    - "#SSN#"
    - "#CardNumber#"
    - "#CarNumber#"
    - "#Email#"

hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
    subdir: ${hydra.job.num}
```

### `conf/model/` — 모델별 config

| 파일 | `model_name` | `architecture` | 비고 |
|------|-------------|----------------|------|
| `kobart.yaml` | `digit82/kobart-summarization` | `bart` | 베이스라인 |
| `kobart_v2.yaml` | `gogamza/kobart-base-v2` | `bart` | 49.48 실측 |
| `kot5.yaml` | `psyche/KoT5-summarization` | `t5` | 49.87 실측, prefix 필요 |
| `pko_t5.yaml` | `paust/pko-t5-large` | `t5` | prefix: `"summarize: "` |
| `solar_qlora.yaml` | `upstage/SOLAR-10.7B-Instruct-v1.0` | `causal_lm` | r=64, alpha=128, bits=4 |

### `conf/training/` — 학습 설정

| 파일 | 주요 변경 | 대상 모델 |
|------|-----------|-----------|
| `baseline.yaml` | 베이스라인 그대로 (lr=1e-5, epoch=20) | KoBART |
| `full.yaml` | lr=3e-5, epoch=50, label_smoothing=0.1 | T5 계열 |
| `qlora.yaml` | lr=2e-5, epoch=5~10, gradient_accum=4 | SOLAR |

### `conf/inference/` — 추론 설정

| 파일 | `num_beams` | 비고 |
|------|------------|------|
| `beam4.yaml` | 4 | 기본 (베이스라인) |
| `beam8.yaml` | 8 | 강화, length_penalty=1.2 |
| `mbr.yaml` | - | MBR decoding, n_samples=10 |

**추론 전략 (ROUGE 향상)**

| 전략 | 설정 | 기대 효과 |
|------|------|-----------|
| Beam width 증가 | beam4 → beam8 | R2/RL 향상 (다양한 후보 탐색) |
| length_penalty | 1.0 → 1.2 | 적절한 길이 요약 유도 |
| MBR decoding | n_samples=10 | ROUGE-L 기준 최적 문장 선택, 최종 제출 후보 |
| max_length_ratio | 0.2 | 대회 규칙(대화 길이 20% 이내)에 맞춤, 기본값 권장 |

- `max_length_ratio=0.0`(기본): 고정 `generate_max_length` 사용. `max_length_ratio=0.2`: 입력 토큰 수의 20% (최소 30토큰).

### 평가 지표 및 목표

#### 대회 공식 ROUGE 정의

본 대회의 최종 점수는 아래와 같이 산출된다.

| 구분 | 설명 |
|------|------|
| **메트릭 단위** | ROUGE-1-F1, ROUGE-2-F1, ROUGE-L-F1 (각각 F1 = 2×P×R/(P+R)) |
| **multi-reference 처리** | 예측 요약 1개 vs 정답 요약 3개 → 각 정답에 대해 R1/R2/RL 점수 산출 후, 메트릭별로 3개 점수의 평균 계산 |
| **최종 점수** | `score_final = mean(R1_F1_ref1..3) + mean(R2_F1_ref1..3) + mean(RL_F1_ref1..3)` |

즉, 세 메트릭의 "3개 정답에 대한 평균값"을 **합산**한 것이 최종 점수이다. (세 평균을 다시 평균 내는 것이 아님)

#### 로컬 평가 (dev.csv) vs 대회 평가

| 항목 | 로컬 (dev.csv) | 대회 (평가 데이터) |
|------|----------------|-------------------|
| 정답 요약 수 | 1개 | 3개 |
| 점수 계산 | `rouge_combined` = R1 + R2 + RL (단일 레퍼런스) | 위 수식대로 3개 레퍼런스 평균 후 합산 |
| WandB 키 | `eval/rouge_1_f1`, `eval/rouge_2_f1`, `eval/rouge_l_f1`, `eval/rouge_combined` | - |
| 체크포인트 선택 | `rouge_combined` 기준 best | - |

- **rouge_combined** (최대 3.0): 로컬에서 체크포인트 선택·early stopping에 사용. 대회 점수의 **하한선/참고치**로 활용.
- 대회 점수는 multi-reference 덕분에 **로컬 dev 점수보다 높게 나올 수 있음**. 목표 "47.12 → 60+"는 대회 스코어 스케일 기준.
- **가설**: dev에서 R2/RL이 상대적으로 높은 모델이 multi-reference 환경에서도 더 잘 일반화할 가능성이 큼.

#### 목표 수치 해석

- 베이스라인 47.12, 목표 60+는 **대회 공식 점수** 기준.
- 로컬 `rouge_combined`는 0~3.0 스케일이므로, 실험 로그 해석 시 혼동하지 않도록 위 정의를 참고한다.

---

## 3. Phase별 구현 계획

---

### Phase 1 — 환경 셋업 & 베이스라인 재현

**목표**: 스킬 구조로 프로젝트 뼈대 구축 + 베이스라인 ROUGE 47.12 재현

#### 구현 범위

| 파일 | 작업 내용 |
|------|-----------|
| `conf/config.yaml` | 메인 config 작성 (위 설계 참고) |
| `conf/model/kobart.yaml` | 베이스라인 모델 config |
| `conf/training/baseline.yaml` | 베이스라인 학습 파라미터 |
| `conf/inference/beam4.yaml` | 기본 추론 설정 |
| `src/data/preprocess.py` | 베이스라인 `Preprocess`, `DatasetForTrain/Val/Inference` 이식 |
| `src/models/summarizer.py` | `bart` 아키텍처 모델 로드 |
| `src/utils/device.py` | **디바이스 자동 감지**: NVIDIA GPU(CUDA) 또는 Mac M4 MPS 우선 사용, 없으면 CPU (train/inference에서 공통 사용) |
| `src/utils/metrics.py` | `compute_metrics` (rouge 라이브러리 기반, 기존 베이스라인 로직) |
| `src/utils/postprocess.py` | 특수 토큰 제거 (베이스라인 기존 로직) |
| `src/train.py` | `@hydra.main` + `Seq2SeqTrainer` + WandB |
| `src/inference.py` | beam search + CSV 저장 |

#### `src/utils/device.py` 설계 (디바이스 자동 감지)

- **목적**: 학습·추론 시 실행 환경에 맞는 디바이스를 자동 선택하여 `train.py`, `inference.py`, `summarizer.py` 등에서 공통 사용.
- **우선순위**: 1) NVIDIA GPU (`torch.cuda.is_available()`) → `cuda`  
  2) Mac M4 등 Apple Silicon MPS (`torch.backends.mps.is_available()`) → `mps`  
  3) 그 외 → `cpu`
- **제공 API**: `get_device()` → `torch.device`, 필요 시 `device_map` 등 Trainer/추론에서 사용할 수 있는 형식으로 반환.
- **주의**: MPS 사용 시 PyTorch 2.0+ 권장; 일부 연산은 MPS 미지원 시 CPU로 fallback 처리 고려.

#### 핵심 결정 사항

- `Preprocess.make_input()`: 베이스라인 로직 유지, 추후 포맷 변경은 config로 제어
- `compute_metrics()`: 현재 `rouge` 라이브러리 사용 유지 (형태소 적용은 Phase 3에서)
- **디바이스**: `src/utils/device.py`의 `get_device()`를 사용해 NVIDIA GPU / Mac M4 MPS 자동 감지, 학습·추론에 일관 적용
- WandB run name: `${model.name}_lr${training.learning_rate}_ep${training.num_train_epochs}` 자동 생성

---

### Phase 2 — 모델 업그레이드

**목표**: 50 → 55~60

#### 구현 범위

| 파일 | 작업 내용 |
|------|-----------|
| `conf/model/kobart_v2.yaml` | `gogamza/kobart-base-v2` config 추가 |
| `conf/model/kot5.yaml` | `psyche/KoT5-summarization` config 추가 |
| `conf/model/pko_t5.yaml` | `paust/pko-t5-large` config 추가 |
| `conf/model/solar_qlora.yaml` | SOLAR QLoRA config 추가 |
| `conf/training/full.yaml` | T5용 학습 설정 (lr=3e-5, epoch=50) |
| `conf/training/qlora.yaml` | SOLAR QLoRA 학습 설정 |
| `src/models/summarizer.py` | `architecture` 분기 추가: `t5`, `causal_lm` |

#### `summarizer.py` 분기 설계

```
architecture = bart   → AutoModelForSeq2SeqLM (기존 방식)
architecture = t5     → AutoModelForSeq2SeqLM + prefix 처리
architecture = causal_lm → AutoModelForCausalLM + peft QLoRA 적용
```

#### 실험 순서 (우선순위)

1. `KoT5-summarization` — 실측 49.87, 즉시 시도 가능
2. `kobart-base-v2` — 실측 49.48, 가벼움
3. `pko-T5-large` — 예상 55+, GPU 비용 있음
4. `SOLAR QLoRA` — 예상 60+, 고사양 GPU 필요

**ROUGE 목표별 모델 우선순위**

| 목표 (대회 점수) | 우선 시도 모델 | 비고 |
|-----------------|----------------|------|
| 50+ | KoT5, kobart_v2 | 단기, 즉시 실험 가능 |
| 55~60 | pko-T5-large | 중기, beam8/MBR 적용 |
| 60+ | SOLAR QLoRA + 고급 디코딩 | 장기, MBR·앙상블 병행 |

---

### Phase 3 — 데이터/학습 전략 고도화

**목표**: 55 → 57~62

#### 구현 범위

| 파일 | 작업 내용 |
|------|-----------|
| `src/data/preprocess.py` | 텍스트 클리닝 함수 추가, 길이 필터, 포맷 변경 옵션 |
| `src/data/augment.py` | back-translation, EDA/AEDA 증강 파이프라인 |
| `src/utils/metrics.py` | 형태소 기반 ROUGE로 교체 (konlpy Okt) |
| `conf/config.yaml` | 누락 special token 9개 추가 |
| `conf/training/full.yaml` | `label_smoothing_factor: 0.1` 추가 |

#### 3-1. 데이터 클리닝 (`preprocess.py` 확장)

- `clean_text()`: 단독 자음/모음, 괄호, 반복 특수기호 제거
- 길이 필터: dialogue 1500자 초과 / summary 50자 미만 or 250자 초과 → IQR 상위 5% drop
- `make_input()` 포맷 옵션: `format: default | prefix_guided` (config으로 제어)

**클리닝·필터 사용 정책**

| Phase | 정책 | 비고 |
|-------|------|------|
| Phase 1~2 | 원본 유지 (클리닝·필터 비활성화) | 베이스라인 재현 시 |
| Phase 3 | `clean_text` + `filter_by_length` 활성화 | dev ROUGE 변화 검증 후 상시 활성화 여부 결정 |
| Phase 3+ | 검증 통과 시 기본값으로 고정 | 이상치 제거로 R2/RL 안정화 기대 |

#### 3-2. 누락 Special Token 추가

```
기존: #Person1~3#, #PhoneNumber#, #Address#, #PassportNumber#
추가: #DateOfBirth#, #SSN#, #CardNumber#, #CarNumber#, #Email#
```
→ `conf/config.yaml`의 `tokenizer.special_tokens`에 통합 관리

- 대회 데이터의 개인정보 마스킹 정책(전화번호, 주소, 이메일 등)과 일치시켜야 함. 모델이 수치를 그대로 생성하지 않고 마스킹 토큰을 사용하면 ROUGE n-gram 매칭에 유리함.

#### 3-3. 한국어 ROUGE (`metrics.py`)

- **Java 불필요**: `korouge-score` (requirements.txt 포함) 사용
- `USE_KOROUGE = True`로 플래그 전환 → 한국어 문자 보존 ROUGE 활성화
- `compare_rouge_modes(preds, refs)` 로 baseline vs korouge 점수 비교 가능
- konlpy Okt(Java 필요) 대신 korouge-score를 표준으로 채택

#### 3-4. 데이터 증강 (`augment.py`)

- `BackTranslationAugmenter`: ko→en→ko 번역 (googletrans 또는 API)
- `EdaAugmenter`: nlpaug 기반 synonym/delete/insert
- 증강 데이터는 ROUGE 필터링 후 `data/train_aug.csv`로 저장
- 최종 제출 전 `train+dev` 합산 학습 스크립트 별도 제공

**증강·TTA 사용 정책**

| 구분 | 정책 | 비고 |
|------|------|------|
| 학습 시 증강 | 원본:증강 비율 1:1 또는 dev ROUGE 기준으로 결정 | 과도한 증강은 품질 저하 가능 → dev로 검증 |
| 추론 시 TTA | `apply_tta()`로 대화 역전 등 N-way 변형 생성 | 다중 요약 후보 중 ROUGE-L 근사 또는 길이 제약으로 최종 선택 |
| TTA 선택 기준 | ROUGE-L 기준 최적 문장 선택 (MBRDecoder와 동일 원리) | Phase 5 앙상블 시 활용 |

---

### Phase 4 — LLM(Solar API) 활용

**목표**: 독립 경로로 55~60 달성, 앙상블 소스로 활용

#### 구현 범위

| 파일 | 작업 내용 |
|------|-----------|
| `src/inference.py` | `SolarAPIInferencer` 클래스 추가 |
| `conf/inference/solar_api.yaml` | Solar API 파라미터 설정 |

#### `src/inference.py` 구조 설계

```
inference.py
├── Seq2SeqInferencer      # beam search (기존 방식)
│   └── run(cfg)
├── MBRInferencer          # MBR decoding
│   └── run(cfg)
└── SolarAPIInferencer     # Solar Chat API
    ├── build_prompt(dialogue, few_shot_examples)
    ├── summarize(dialogue) → str
    └── run(cfg)           # rate limit 처리 포함
```

#### Solar API 프롬프트 전략

- `conf/inference/solar_api.yaml`에 `prompt_style: zero_shot | few_shot | chain_of_thought` 옵션
- few-shot example 선택: BM25 유사도 기반 (train 데이터에서 동적 선택)
- 생성 파라미터: `temperature=0.1`, `top_p=0.9`, `max_tokens=150`

---

### Phase 5 — 앙상블 & 후처리

**목표**: 60 → 최상위권

#### 구현 범위

| 파일 | 작업 내용 |
|------|-----------|
| `src/utils/postprocess.py` | 후처리 파이프라인 완성 |
| `src/inference.py` | MBR decoding, TTA 구현 |
| `src/ensemble.py` | GroupKFold OOF, 가중치 앙상블 |

#### `src/ensemble.py` 설계

```
ensemble.py
├── GroupKFoldTrainer      # topic 그룹 기반 5-fold CV
│   └── train_oof(cfg)
├── WeightedEnsemble       # SOLAR(0.5) + KoT5(0.3) + KoBART(0.2)
│   └── predict(predictions_list, weights)
└── MBRDecoder             # N개 후보 중 평균 ROUGE 최고 선택
    └── decode(candidates)
```

#### 후처리 파이프라인 (`postprocess.py`)

1. 특수 토큰 제거 (`<s>`, `</s>`, `<pad>`, `<usr>`)
2. 과도한 공백 정리
3. 문장 끝 마침표 보장
4. 반복 문장 제거
5. 최소 길이 보장 (10자 미만 → 재생성 플래그)

---

## 4. 전체 체크리스트

> **범례**: ✅ 구현+단위테스트 통과 / ⚠️ 구현됨(환경 제약) / 🔲 학습/API 실행 필요

### Phase 1 — 환경 셋업 & 베이스라인

**환경 설정**
- [x] `.env` 파일 API key 설정 완료 (`WANDB_API_KEY`, `HF_TOKEN`, `UPSTAGE_API_KEY`) ✅
- [x] `requirements.txt` 기준 패키지 설치 확인 ✅
- [x] `data/` 디렉토리에 `train.csv`(12457건), `dev.csv`(499건), `test.csv`(499건) 배치 ✅
- [x] GPU/MPS 확인: Apple Silicon MPS 감지 확인 (`device=mps`) ✅

**스킬 구조 구축**
- [x] `conf/` 디렉토리 구조 생성 (model/, training/, inference/) ✅
- [x] `src/` 디렉토리 구조 생성 (data/, models/, utils/) ✅
- [x] `prediction/`, `checkpoints/` 디렉토리 생성 ✅

**베이스라인 구현**
- [x] `conf/config.yaml` 작성 (special token 11개 = 베이스라인 6개 + 추가 5개) ✅
- [x] `conf/model/kobart.yaml` 작성 ✅
- [x] `conf/training/baseline.yaml` 작성 ✅
- [x] `conf/inference/beam4.yaml` 작성 ✅
- [x] `src/data/preprocess.py` — `Preprocess`, `DatasetForTrain/Val/Inference` 구현 ✅
- [x] `src/models/summarizer.py` — `bart` 아키텍처 로드 ✅
- [x] `src/utils/device.py` — NVIDIA GPU / Mac M4 MPS 자동 감지, `get_device()` 구현 ✅
- [x] `src/utils/metrics.py` — `compute_metrics` 구현 ✅
- [x] `src/utils/postprocess.py` — 특수 토큰 제거 구현 ✅
- [x] `src/train.py` — `@hydra.main` + `Seq2SeqTrainer` + WandB (device는 `device.py` 사용) ✅
- [x] `src/inference.py` — beam search + CSV 출력 (device는 `device.py` 사용) ✅

**베이스라인 검증** *(GPU 학습 실행 필요)*
- [ ] `python src/train.py` 실행 → 오류 없이 학습 시작 확인 🔲
- [ ] WandB 대시보드에 run 생성 확인 🔲
- [ ] Dev ROUGE 점수 기록 (목표: 47.12 ± 1) 🔲
- [ ] `python src/inference.py` 실행 → `prediction/output.csv` 생성 확인 🔲
- [ ] Hydra `outputs/` 디렉토리에 config 자동 저장 확인 🔲

---

### Phase 2 — 모델 업그레이드

**Config 추가**
- [x] `conf/model/kobart_v2.yaml` 작성 (gogamza/kobart-base-v2) ✅
- [x] `conf/model/kot5.yaml` 작성 (prefix: `"summarize: "`) ✅
- [x] `conf/model/pko_t5.yaml` 작성 ✅
- [x] `conf/model/solar_qlora.yaml` 작성 (r=64, alpha=128, bits=4) ✅
- [x] `conf/training/full.yaml` 작성 (lr=3e-5, epoch=50, label_smoothing=0.1) ✅
- [x] `conf/training/qlora.yaml` 작성 (epoch=5, gradient_accum=4, bf16) ✅

**summarizer.py 확장**
- [x] `architecture: t5` 분기 구현 (prefix 처리 포함) — KoT5 로드 테스트 통과 ✅
- [x] `architecture: causal_lm` 분기 구현 (peft QLoRA) ✅
- [x] T5 모델에서 `return_token_type_ids=False` 유지 확인 ✅

**모델 실험** *(GPU 학습 실행 필요)*
- [ ] KoT5-summarization 학습 & 평가 → Dev ROUGE 기록 🔲
- [ ] kobart-base-v2 학습 & 평가 → Dev ROUGE 기록 🔲
- [ ] pko-T5-large 학습 & 평가 → Dev ROUGE 기록 🔲
- [ ] (선택) SOLAR QLoRA 학습 & 평가 → Dev ROUGE 기록 🔲
- [ ] Hydra sweep: `python src/train.py -m model=kobart,kot5,pko_t5` 🔲

---

### Phase 3 — 데이터/학습 전략 고도화

**데이터 클리닝**
- [x] `clean_text()` 함수 구현 (자음/모음, 괄호, 반복 특수기호 제거) ✅
  - `ㅋㅋㅋ 안녕하세요` → `안녕하세요` 확인
  - `#Person1#` 태그 보존 확인
- [x] 길이 필터 구현 (`filter_by_length`: dialogue≤1500, summary 50~250) ✅
- [x] 클리닝 전/후 데이터 통계 비교 — train 12457 → 필터 후 11117건 (1340건 제거) ✅

**Special Token**
- [x] `conf/config.yaml`에 누락 5개 토큰 추가 확인 (총 11개) ✅
- [x] `tokenizer.add_special_tokens()` 후 `resize_token_embeddings()` 호출 확인 — vocab 30000 → 30011 ✅

**한국어 ROUGE (Java 불필요)**
- [x] `korouge-score` 기반 한국어 ROUGE 구현 — Java/konlpy 없이 동작 ✅
  - `USE_KOROUGE=False` → `rouge` 라이브러리 (Phase 1~2 기본, 베이스라인 호환)
  - `USE_KOROUGE=True`  → `korouge-score` (Phase 3+ 권장, 한국어 문자 보존)
- [x] `compare_rouge_modes()` 함수로 두 모드 점수 비교 가능 ✅
- [x] 두 모드 점수 차이 실측 — dev 20건 기준 rouge-1 차이 ≈ ±0.004 ✅

**데이터 증강**
- [x] `src/data/augment.py` 기본 구조 작성 ✅
- [x] Back-translation 파이프라인 구현 (`BackTranslationAugmenter`) ✅
- [x] EDA/AEDA (nlpaug) 구현 및 증강 데이터 품질 확인 — `EdaAugmenter` 동작 확인 ✅
- [x] 증강 데이터 ROUGE 필터링 적용 (`augment_dataset` 내 threshold 필터) ✅
- [ ] `data/train_aug.csv` 생성 🔲 (googletrans API 실행 필요)

**학습 전략**
- [x] `label_smoothing_factor=0.1` — `full.yaml`에 설정, config 로드 확인 ✅
- [x] `conf/inference/beam8.yaml` 작성 (length_penalty=1.2) ✅
- [ ] Train+Dev 합산 학습 실험 (최종 제출 전) 🔲

**학습 하이퍼파라미터 튜닝 (ROUGE 향상 중심)**

| 축 | 기본값 | 탐색 범위 | 비고 |
|----|--------|-----------|------|
| learning_rate | 1e-5 (baseline), 3e-5 (full) | 1e-5 ~ 5e-5 | T5 계열은 3e-5 권장 |
| num_train_epochs | 20 (baseline), 50 (full) | 15 ~ 50 | early stopping으로 조기 종료 |
| label_smoothing_factor | 0.1 | 0.05 ~ 0.15 | R2/RL 안정화에 기여 |
| warmup_ratio | 0.1 | 0.05 ~ 0.15 | 학습 초기 안정화 |
| per_device_train_batch_size | 8 | 4 ~ 16 | GPU 메모리에 따라 |

- Early stopping 기준: `rouge_combined`. 세 메트릭 균형이 목표이므로 R1만 높은 모델보다 R2/RL도 함께 높은 모델을 선호.

---

### Phase 4 — LLM 활용

**Solar API**
- [x] `.env`의 `UPSTAGE_API_KEY` 설정 확인 ✅
- [x] `conf/inference/solar_api.yaml` 작성 ✅
- [x] `src/inference.py`에 `SolarAPIInferencer` 클래스 추가 ✅
- [ ] zero-shot 프롬프트 구현 및 dev 100개 테스트 🔲 (API 실행 필요)
- [x] few-shot (3-shot) 프롬프트 구현 (`build_prompt()`) ✅
- [x] BM25 기반 few-shot 예제 선택 구현 (`_load_few_shot_examples()`) ✅
- [x] rate limit 처리 확인 (RPM 기반 delay 계산 구현) ✅
- [ ] `prediction/output_solar.csv` 생성 확인 🔲 (API 실행 필요)

---

### Phase 5 — 앙상블 & 후처리

**후처리**
- [x] `postprocess.py` 5단계 파이프라인 완성 ✅
  1. 특수 토큰 제거 ✅
  2. 과도한 공백 정리 ✅
  3. 문장 끝 마침표 보장 ✅
  4. 반복 문장 제거 ✅
  5. (최소 길이 보장은 재생성 플래그로 향후 추가)
- [ ] 후처리 전/후 Dev ROUGE 변화 확인 🔲

**MBR Decoding**
- [x] `MBRInferencer` 구현 (`Seq2SeqInferencer` 내 `do_sample=True` 모드, `_mbr_select()`) ✅
- [ ] beam4 vs beam8 vs MBR 성능 비교 🔲

**앙상블**
- [x] `src/ensemble.py` 작성 ✅
- [x] `GroupKFoldTrainer` 구현 (topic 그룹, n_splits=5) ✅
- [ ] OOF 예측 저장 및 검증 ROUGE 계산 🔲
- [x] `WeightedEnsemble` 구현 (가중치: 명시적 or OOF 기반 자동 계산) ✅
- [ ] SOLAR + KoT5 + KoBART 앙상블 최종 예측 생성 🔲

**TTA**
- [x] 발화 순서 역전 augmentation 구현 (`reverse_utterances()`, `apply_tta()` in preprocess.py) ✅
- [ ] 8-way TTA → ROUGE 투표 방식 검증 🔲

---

## 5. 테스트 계획

### 5-1. 단위 테스트 (구현 직후 확인)

| 테스트 항목 | 확인 방법 | 기대 결과 |
|-------------|-----------|-----------|
| `get_device()` | `from src.utils.device import get_device; get_device()` | 환경에 따라 `cuda` / `mps` / `cpu` 중 하나, `torch.device` 반환 |
| `Preprocess.make_input()` train 모드 | 반환값 3개 (encoder, decoder_in, decoder_out) | 길이 동일, BOS/EOS 붙어있음 |
| `Preprocess.make_input()` test 모드 | 반환값 2개 (encoder, decoder_in) | decoder_in이 모두 `<s>` |
| `clean_text()` | 단독 자음 포함 입력 → 출력 확인 | 자음 제거됨 |
| `compute_metrics()` 형태소 ROUGE | 동일 문장 입력 → ROUGE=1.0 | 1.0 반환 |
| Special token 추가 | `tokenizer.vocab_size` 증가 확인 | +9개 |
| `postprocess()` | 특수 토큰 포함 입력 → 제거 확인 | 토큰 없음 |
| T5 prefix | `architecture=t5`로 로드 시 `"summarize: "` prefix 붙음 | 인코더 입력 확인 |
| QLoRA 로드 | `architecture=causal_lm` + bits=4 로드 | 메모리 사용량 감소 확인 |

### 5-2. 통합 테스트 (Phase 완료 시점)

| Phase | 테스트 | 합격 기준 |
|-------|--------|-----------|
| Phase 1 완료 | `python src/train.py` 전체 실행 | Dev ROUGE ≥ 46 (재현) |
| Phase 1 완료 | `python src/inference.py` 실행 | `prediction/output.csv` 정상 생성 |
| Phase 2 완료 | `python src/train.py model=kot5` | Dev ROUGE ≥ 49 |
| Phase 2 완료 | Hydra sweep 3모델 비교 | WandB에 3개 run 생성, 점수 비교 가능 |
| Phase 3 완료 | 클리닝 적용 후 재학습 | Dev ROUGE ≥ +1 향상 |
| Phase 3 완료 | 형태소 ROUGE 적용 | 평가 점수가 대회 점수와 근사 |
| Phase 4 완료 | Solar API 전체 dev 추론 | Dev ROUGE ≥ 50 |
| Phase 5 완료 | 앙상블 최종 예측 | Dev ROUGE ≥ 58 |

### 5-3. ROUGE 측정 기준

모든 Dev 평가는 아래 방식으로 통일:

```
형태소 기반 ROUGE (konlpy Okt)
Score = ROUGE-1-F1 + ROUGE-2-F1 + ROUGE-L-F1
```

- 베이스라인 기존 `rouge` 라이브러리와 점수가 다를 수 있음 → Phase 3 이후 형태소 기반으로 통일
- Phase 1~2는 기존 `rouge` 라이브러리 사용 허용 (속도 우선)

### 5-4. 실험 기록 양식 (WandB + 로컬)

매 실험마다 아래를 기록:

```
실험명:
모델:
config: model=X training=Y inference=Z
lr / batch / epoch / beams:
특수사항 (클리닝 여부, 증강 여부 등):
Dev ROUGE-1 / ROUGE-2 / ROUGE-L / Total:
Public Test ROUGE-F1:
비고:
```

---

## 6. 완료 기준

| Phase | Dev ROUGE 목표 | 완료 조건 |
|-------|---------------|-----------|
| Phase 1 | ≥ 46 | 베이스라인 재현, 파이프라인 정상 동작 |
| Phase 2 | ≥ 52 | KoT5 또는 pko-T5 fine-tuning 성공 |
| Phase 3 | ≥ 56 | 클리닝 + 형태소 ROUGE + special token 적용 |
| Phase 4 | ≥ 54 (Solar 독립) | Solar API few-shot 추론 완성 |
| Phase 5 | ≥ 60 | 앙상블 최종 예측 완성, submission 제출 |

### 최종 제출 체크리스트

- [ ] 형태소 기반 Dev ROUGE ≥ 58 이상 모델 확보
- [ ] Train+Dev 합산 재학습 완료
- [ ] 후처리 파이프라인 적용
- [ ] `prediction/output_final.csv` 컬럼 형식 확인 (`fname`, `summary`)
- [ ] Public leaderboard 제출 및 점수 확인
- [ ] Private 결과 대비 LB shake-up 여부 점검 (GroupKFold OOF 점수 비교)

### 제출 전 검증 플로우

| 단계 | 확인 항목 |
|------|-----------|
| 1 | dev `rouge_combined` 기준 최소 허들 충족 (예: 0.75 이상) |
| 2 | 예측 요약 샘플 인지적 검토: 핵심 정보 보존, 개체명 유지, 대화 길이 20% 이내 |
| 3 | dev 점수 상승과 직관적 요약 품질을 함께 확인 (과적합 방지) |

### 위험 요소 및 주의사항

- **과적합**: dev에만 맞춘 과도한 튜닝은 대회 공개/비공개 스플릿에서 성능 저하를 가져올 수 있음. dev 점수 상승 + 직관적 요약 품질 확인을 병행할 것.
- **한국어 ROUGE (korouge-score)**: 토크나이저/형태소 분석기 버전에 따라 점수가 변동될 수 있으므로, 버전 고정을 권장.
