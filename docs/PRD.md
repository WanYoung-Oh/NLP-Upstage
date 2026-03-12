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

wandb:
  entity: "${oc.env:WANDB_ENTITY}"
  project: "${oc.env:WANDB_PROJECT}"
  name: "${model.name}_lr${training.learning_rate}_ep${training.num_train_epochs}"

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
| `src/utils/metrics.py` | `compute_metrics` (rouge 라이브러리 기반, 기존 베이스라인 로직) |
| `src/utils/postprocess.py` | 특수 토큰 제거 (베이스라인 기존 로직) |
| `src/train.py` | `@hydra.main` + `Seq2SeqTrainer` + WandB |
| `src/inference.py` | beam search + CSV 저장 |

#### 핵심 결정 사항

- `Preprocess.make_input()`: 베이스라인 로직 유지, 추후 포맷 변경은 config로 제어
- `compute_metrics()`: 현재 `rouge` 라이브러리 사용 유지 (형태소 적용은 Phase 3에서)
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

#### 3-2. 누락 Special Token 추가

```
기존: #Person1~3#, #PhoneNumber#, #Address#, #PassportNumber#
추가: #DateOfBirth#, #SSN#, #CardNumber#, #CarNumber#, #Email#
```
→ `conf/config.yaml`의 `tokenizer.special_tokens`에 통합 관리

#### 3-3. 형태소 기반 ROUGE (`metrics.py`)

- `konlpy.tag.Okt`로 형태소 분석 후 ROUGE 계산
- 대회 공식 평가 방식과 동일 → dev 점수 신뢰도 향상

#### 3-4. 데이터 증강 (`augment.py`)

- `BackTranslationAugmenter`: ko→en→ko 번역 (googletrans 또는 API)
- `EdaAugmenter`: nlpaug 기반 synonym/delete/insert
- 증강 데이터는 ROUGE 필터링 후 `data/train_aug.csv`로 저장
- 최종 제출 전 `train+dev` 합산 학습 스크립트 별도 제공

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

### Phase 1 — 환경 셋업 & 베이스라인

**환경 설정**
- [ ] `.env` 파일 API key 설정 완료 (`WANDB_API_KEY`, `HF_TOKEN`, `UPSTAGE_API_KEY`)
- [ ] `requirements.txt` 기준 패키지 설치 확인
- [ ] `data/` 디렉토리에 `train.csv`, `dev.csv`, `test.csv` 배치
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` GPU 확인

**스킬 구조 구축**
- [ ] `conf/` 디렉토리 구조 생성 (model/, training/, inference/)
- [ ] `src/` 디렉토리 구조 생성 (data/, models/, utils/)
- [ ] `prediction/`, `checkpoints/` 디렉토리 생성

**베이스라인 구현**
- [ ] `conf/config.yaml` 작성 (누락 special token 9개 포함)
- [ ] `conf/model/kobart.yaml` 작성
- [ ] `conf/training/baseline.yaml` 작성
- [ ] `conf/inference/beam4.yaml` 작성
- [ ] `src/data/preprocess.py` — `Preprocess`, `DatasetForTrain/Val/Inference` 구현
- [ ] `src/models/summarizer.py` — `bart` 아키텍처 로드
- [ ] `src/utils/metrics.py` — `compute_metrics` 구현
- [ ] `src/utils/postprocess.py` — 특수 토큰 제거 구현
- [ ] `src/train.py` — `@hydra.main` + `Seq2SeqTrainer` + WandB
- [ ] `src/inference.py` — beam search + CSV 출력

**베이스라인 검증**
- [ ] `python src/train.py` 실행 → 오류 없이 학습 시작 확인
- [ ] WandB 대시보드에 run 생성 확인
- [ ] Dev ROUGE 점수 기록 (목표: 47.12 ± 1)
- [ ] `python src/inference.py` 실행 → `prediction/output.csv` 생성 확인
- [ ] Hydra `outputs/` 디렉토리에 config 자동 저장 확인

---

### Phase 2 — 모델 업그레이드

**Config 추가**
- [ ] `conf/model/kobart_v2.yaml` 작성
- [ ] `conf/model/kot5.yaml` 작성 (prefix: `"summarize: "`)
- [ ] `conf/model/pko_t5.yaml` 작성
- [ ] `conf/model/solar_qlora.yaml` 작성 (r=64, alpha=128, bits=4)
- [ ] `conf/training/full.yaml` 작성 (lr=3e-5, epoch=50)
- [ ] `conf/training/qlora.yaml` 작성 (epoch=5~10, gradient_accum=4)

**summarizer.py 확장**
- [ ] `architecture: t5` 분기 구현 (prefix 처리 포함)
- [ ] `architecture: causal_lm` 분기 구현 (peft QLoRA)
- [ ] T5 모델에서 `return_token_type_ids=False` 유지 확인

**모델 실험**
- [ ] KoT5-summarization 학습 & 평가 → Dev ROUGE 기록
- [ ] kobart-base-v2 학습 & 평가 → Dev ROUGE 기록
- [ ] pko-T5-large 학습 & 평가 → Dev ROUGE 기록
- [ ] (선택) SOLAR QLoRA 학습 & 평가 → Dev ROUGE 기록
- [ ] Hydra sweep: `python src/train.py -m model=kobart,kot5,pko_t5`

---

### Phase 3 — 데이터/학습 전략 고도화

**데이터 클리닝**
- [ ] `clean_text()` 함수 구현 (자음/모음, 괄호, 반복 특수기호 제거)
- [ ] 길이 필터 구현 (dialogue <1500, summary 50~250)
- [ ] 클리닝 전/후 데이터 통계 비교 (건수, 평균 길이)

**Special Token**
- [ ] `conf/config.yaml`에 누락 9개 토큰 추가 확인
- [ ] `tokenizer.add_special_tokens()` 후 `resize_token_embeddings()` 호출 확인

**형태소 기반 ROUGE**
- [ ] konlpy Okt 설치 및 동작 확인
- [ ] `metrics.py`에 Okt 기반 토크나이징 적용
- [ ] 형태소 ROUGE vs 기존 ROUGE 점수 차이 확인 및 기록

**데이터 증강**
- [ ] `src/data/augment.py` 기본 구조 작성
- [ ] Back-translation 파이프라인 구현 및 샘플 100개 테스트
- [ ] EDA/AEDA (nlpaug) 구현 및 증강 데이터 품질 확인
- [ ] 증강 데이터 ROUGE 필터링 적용
- [ ] `data/train_aug.csv` 생성

**학습 전략**
- [ ] `label_smoothing_factor=0.1` 적용 실험
- [ ] `conf/inference/beam8.yaml` 작성 (length_penalty=1.2)
- [ ] Train+Dev 합산 학습 실험 (최종 제출 전)

---

### Phase 4 — LLM 활용

**Solar API**
- [ ] `.env`의 `UPSTAGE_API_KEY` 설정 확인
- [ ] `conf/inference/solar_api.yaml` 작성
- [ ] `src/inference.py`에 `SolarAPIInferencer` 클래스 추가
- [ ] zero-shot 프롬프트 구현 및 dev 100개 테스트
- [ ] few-shot (3-shot) 프롬프트 구현 및 성능 비교
- [ ] BM25 기반 few-shot 예제 선택 구현 (선택)
- [ ] rate limit 처리 확인 (분당 100건)
- [ ] `prediction/output_solar.csv` 생성 확인

---

### Phase 5 — 앙상블 & 후처리

**후처리**
- [ ] `postprocess.py` 5단계 파이프라인 완성
- [ ] 후처리 전/후 Dev ROUGE 변화 확인

**MBR Decoding**
- [ ] `MBRInferencer` 구현 (n_samples=10, temperature sampling)
- [ ] beam4 vs beam8 vs MBR 성능 비교

**앙상블**
- [ ] `src/ensemble.py` 작성
- [ ] `GroupKFoldTrainer` 구현 (topic 그룹, n_splits=5)
- [ ] OOF 예측 저장 및 검증 ROUGE 계산
- [ ] `WeightedEnsemble` 구현 (가중치: OOF 기반 자동 계산)
- [ ] SOLAR + KoT5 + KoBART 앙상블 최종 예측 생성

**TTA**
- [ ] 발화 순서 역전 augmentation 구현
- [ ] 8-way TTA → ROUGE 투표 방식 검증

---

## 5. 테스트 계획

### 5-1. 단위 테스트 (구현 직후 확인)

| 테스트 항목 | 확인 방법 | 기대 결과 |
|-------------|-----------|-----------|
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
