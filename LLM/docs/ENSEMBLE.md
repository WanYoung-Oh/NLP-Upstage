# MBR 멀티-체크포인트 앙상블 구현 계획서

> 현재 리더보드 점수: **52.0083** | 목표: **54.0+**
> 최초 작성: 2026-03-28 | 최종 수정: 2026-03-28

---

## 목차

1. [현황 분석 및 문제 정의](#1-현황-분석-및-문제-정의)
2. [이론적 배경 및 연구 근거](#2-이론적-배경-및-연구-근거)
3. [구현 설계: Mode 1 — 체크포인트 MBR](#3-구현-설계-mode-1--체크포인트-mbr)
4. [구현 설계: Mode 2 — 체크포인트 × 프롬프트 MBR](#4-구현-설계-mode-2--체크포인트--프롬프트-mbr)
5. [구현 설계: Mode 3 — Sampling 기반 다양성](#5-구현-설계-mode-3--sampling-기반-다양성)
6. [비대칭 MBR (Asymmetric MBR)](#6-비대칭-mbr-asymmetric-mbr)
7. [YAML 설정 관리 (conf/)](#7-yaml-설정-관리-conf)
8. [CLI 명세](#8-cli-명세)
9. [성능 향상 가능성 평가](#9-성능-향상-가능성-평가)
10. [구현 로드맵](#10-구현-로드맵)
11. [주의사항 및 리스크](#11-주의사항-및-리스크)

---

## 1. 현황 분석 및 문제 정의

### 1.1 현재 파이프라인

```
단일 체크포인트 (r4b_response_only_ckpt)
    └── 7개 프롬프트 변형 병렬 생성 (greedy decoding)
            └── ROUGE-1 기반 MBR 선택
                    └── 최종 요약 출력  →  LB 52.0083
```

**현재 MBR의 한계:**
- 후보 풀이 단일 모델의 분포에만 의존 → **다양성 천장**이 존재
- Greedy 디코딩만 사용 → 같은 프롬프트에서 항상 동일한 출력 → 확률적 다양성 없음
- ROUGE-1만 사용한 MBR utility → **메트릭 편향(metric bias)** 발생 가능

### 1.2 리더보드 확인된 상위 체크포인트

리더보드 제출 결과로 확인된 상위 3개 체크포인트. **이 3개만 앙상블에 사용한다.**

| 이름 | 경로 | 훈련 방식 | LB 확인 | 비고 |
|---|---|---|---|---|
| `r4b_response_only` | `response_only_SFT/r4b_response_only_ckpt/` | Response-only SFT | ✅ | 현재 메인, MBR 시 0.5716 |
| `qwen14b_lora_v1` | `response_only_SFT/outputs/qwen3_14b_lora_sft/lora_adapter/` | Response-only SFT | ✅ | — |
| `exp_B_14b` | `response_only_SFT/outputs/exp_B_r32_a64_lr1e4/lora_adapter/` | SFT (r32, α64, lr1e-4) | ✅ | — |

> 이하 미확인 체크포인트(simpo_v2, baseline_14b, q35 계열)는 conf/checkpoints.yaml에 등록은 유지하되
> 앙상블 설정 파일에서는 제외한다. 품질이 불확실한 체크포인트를 포함하면
> utility 계산이 오염될 수 있기 때문이다. (→ [6절 비대칭 MBR](#6-비대칭-mbr-asymmetric-mbr) 참조)

---

## 2. 이론적 배경 및 연구 근거

### 2.1 MBR 디코딩 원리

$$y^* = \arg\max_{y \in \mathcal{H}} \frac{1}{|\mathcal{R}|} \sum_{r \in \mathcal{R}} U(y, r)$$

- $\mathcal{H}$: hypothesis pool (후보 텍스트 집합)
- $\mathcal{R}$: reference pool (평가 기준 집합)
- $U$: utility 함수 (ROUGE, BERTScore 등)

표준 MBR에서는 $\mathcal{H} = \mathcal{R}$ (동일 집합). **비대칭 MBR에서는 $\mathcal{H} \supseteq \mathcal{R}$** 로 분리 가능.

**핵심 원리:** $\mathcal{H}$와 $\mathcal{R}$의 **다양성(diversity)** 이 MBR 성능의 천장을 결정.

> *"Theoretical Aspects of Bias and Diversity in Minimum Bayes Risk Decoding"* (arXiv:2410.15021, 2024)
> 다양성과 편향은 반비례 관계. 다양한 pseudo-reference가 utility 추정의 분산을 줄인다.

### 2.2 멀티 프롬프트 MBR의 유효성

> *"Improving Minimum Bayes Risk Decoding with Multi-Prompt"* (EMNLP 2024, PMC:12226151)
> - 단일 프롬프트 대비 멀티 프롬프트 MBR이 **모든 태스크**에서 일관되게 우수
> - Text Simplification: 74.67 → **79.08 LENS** (+4.41)
> - Machine Translation: 88.93 → **90.36 COMET** (+1.43)
> - **최대 64개 프롬프트**까지 추가 이득, 이후 수확 체감

**현재 구현의 위치:** 7개 프롬프트 MBR (단일 체크포인트) → 이 논문의 권고안 부분 구현 중.

### 2.3 멀티 체크포인트 앙상블

> *"CUED at ProbSum 2023: Hierarchical Ensemble of Summarization Models"* (ACL BioNLP 2023)
> - 다양한 체크포인트 → output-level 앙상블 → MBR 적용
> - ProbSum 2023 **1위** (ROUGE-L 32.77)

**앙상블 레벨 비교:**

| 레벨 | 방식 | 구현 난이도 | 예상 향상 |
|---|---|---|---|
| Output-level | 각 체크포인트 출력 텍스트를 MBR로 선택 | 쉬움 | +0.5~2.0 ROUGE-1 |
| Token-level | softmax 확률분포를 체크포인트 간 평균 후 생성 | 어려움 | +2.0~4.0 ROUGE-1 |

> 모든 체크포인트가 동일한 Qwen3-14B 베이스 + LoRA 구조이므로 **output-level**이 현실적 선택.

### 2.4 한국어 형태소 분석 최적화 (필수)

MBR utility 계산에 앞서 **한국어 형태소 분석**이 선행되어야 한다.
한국어는 조사(은/는/이/가/을/를)와 어미(-았/었/겠/겠다)가 단어에 붙어 쓰이는 교착어 특성상,
공백 기반 토크나이징으로는 ROUGE를 올바르게 계산할 수 없다.

```
공백 분리:  "Person1이 회의를 제안했다"  →  ["Person1이", "회의를", "제안했다"]
MeCab 분리: "Person1이 회의를 제안했다"  →  ["Person1", "이", "회의", "를", "제안", "했", "다"]
```

공백 분리 방식은 "Person1이"와 "Person1은"을 완전히 다른 토큰으로 처리하여
실제로 같은 내용인데도 ROUGE 점수가 낮게 나온다.

**적용 지점:**
- MBR utility 계산 시 모든 후보 텍스트를 MeCab으로 형태소 분리 후 ROUGE 계산
- 대회 평가 기준과 동일한 방식 유지 → 내부 dev 점수와 리더보드 점수의 괴리 최소화
- 모든 YAML 설정에서 `use_mecab: true` 기본값 유지

**MeCab 미설치 시 폴백:** 기존 코드(`mecab_ko.py`)에서 공백 기반 토크나이징으로 자동 폴백.
단, 이 경우 MBR 선택 품질이 저하되므로 환경에 MeCab 설치를 강력 권장.

```bash
# MeCab 설치 확인
python -c "import MeCab; m = MeCab.Tagger(); print('MeCab OK')"
```

### 2.5 MBR Utility 함수 선택

> *"High Quality Rather than High Model Probability: MBR with Neural Metrics"* (TACL 2022)
> *"Mitigating Metric Bias in Minimum Bayes Risk Decoding"* (WMT 2024)

| Utility 함수 | 한국어 품질 | 속도 | 메트릭 편향 | 권장도 |
|---|---|---|---|---|
| ROUGE-1 (현재) | 보통 | 빠름 | 높음 | ★★★ |
| ROUGE-1+2+L 평균 | 보통 | 빠름 | 중간 | ★★★★ |
| BERTScore (klue/roberta) | 우수 | 느림 | 낮음 | ★★★★★ |
| BERTScore + ROUGE 복합 | 최우수 | 느림 | 최저 | ★★★★★ |

> BERTScore 사용 시: `klue/roberta-base`는 한국어 조사·어미 변화에 강건한 의미 유사도를 제공.
> ROUGE와 병행 시 각각 MeCab 토크나이징(ROUGE) + klue/roberta 토크나이저(BERTScore)로 이중 적용.

### 2.6 세 가지 다양성 축

```
다양성 축 1: 프롬프트 다양성   (surface framing 변화)
다양성 축 2: 체크포인트 다양성 (learned weight distribution 변화)
다양성 축 3: Sampling 다양성   (확률적 탐색, 동일 모델 내 다양한 표현)
```

이 세 축은 서로 **직교**하므로, 결합 시 가산적 효과가 기대된다.

```
Mode 1: 3 ckpt × 1 prompt (greedy)              = 3 후보
Mode 2: 3 ckpt × 7 prompts (greedy)             = 21 후보
Mode 3: 3 ckpt × 1 prompt × K samples           = 3K 후보
최대:   3 ckpt × 7 prompts × K samples          = 21K 후보
```

---

## 3. 구현 설계: Mode 1 — 체크포인트 MBR

### 3.1 개요

상위 3개 체크포인트에서 대표 프롬프트 1개씩 생성, 3 후보로 MBR 선택.
가장 빠른 검증용. 체크포인트 다양성 효과만 단독으로 측정할 수 있다.

```
r4b_response_only  × qa_style  →  요약_A
qwen14b_lora_v1    × base      →  요약_B  →  MBR (3 후보)  →  최종 요약
exp_B_14b          × qa_style  →  요약_C
```

### 3.2 실행 흐름

```python
for ckpt in config.checkpoints:           # 순차 로딩 (VRAM 관리)
    model = load_lora_model(ckpt.path)
    predictions[ckpt.name] = generate(model, test_df, prompt=ckpt.prompt)
    del model; torch.cuda.empty_cache()

final = apply_mbr(predictions, utility=config.mbr.utility)
```

### 3.3 특징

- 후보 수: **3개** (최소 구성)
- 소요 시간: **~38분** (14B × 3 체크포인트)
- 용도: 체크포인트 다양성 효과 단독 검증, Mode 2 진입 전 빠른 확인

---

## 4. 구현 설계: Mode 2 — 체크포인트 × 프롬프트 MBR

### 4.1 개요

상위 3개 체크포인트 × 7개 프롬프트의 전체 조합으로 21 후보 풀 구성 후 MBR 선택.
체크포인트 다양성 + 프롬프트 다양성을 동시에 활용하는 **핵심 전략**.

```
r4b_response_only × [base, qa_style, topic, narrative, observer, gold_mimic, length_constrained]
qwen14b_lora_v1   × [base, qa_style, topic, narrative, observer, gold_mimic, length_constrained]
exp_B_14b         × [base, qa_style, topic, narrative, observer, gold_mimic, length_constrained]
                                                                        ↓
                                                              총 21 후보 → MBR
```

### 4.2 실행 흐름

```python
all_predictions = {}  # key: "{ckpt_name}__{prompt_name}"

for ckpt in config.checkpoints:
    model = load_lora_model(ckpt.path)
    for prompt_name in config.prompts:
        key = f"{ckpt.name}__{prompt_name}"
        all_predictions[key] = generate(model, test_df, prompt=prompt_name)
    del model; torch.cuda.empty_cache()

final = apply_mbr(all_predictions, utility=config.mbr.utility)
```

### 4.3 후보 수 vs 소요 시간

| 구성 | 후보 수 | 예상 시간 | 비고 |
|---|---|---|---|
| 3 ckpt × 3 prompts | 9 | ~1.1h | 빠른 검증 |
| 3 ckpt × 7 prompts | **21** | **~2.6h** | **권장** |
| 3 ckpt × 7 prompts + sampling | 21~40+ | ~4h+ | Mode 2 + 3 결합 |

### 4.4 체크포인트 가중치 적용

리더보드 점수 또는 dev ROUGE 기준으로 상위 체크포인트에 높은 가중치를 부여하면
MBR 선택 시 더 신뢰도 높은 체크포인트의 평가가 더 많이 반영된다.

```yaml
checkpoints:
  - name: r4b_response_only
    weight: 1.2           # dev ROUGE 기준 조정
  - name: qwen14b_lora_v1
    weight: 1.5           # 상위 성능 시 높은 가중치
  - name: exp_B_14b
    weight: 1.0
```

---

## 5. 구현 설계: Mode 3 — Sampling 기반 다양성

### 5.1 배경 및 동기

Mode 1/2는 모두 **greedy decoding** 기반이다. 동일 체크포인트 + 동일 프롬프트 조합에서
greedy 결과는 항상 동일하므로, 후보 풀의 다양성이 체크포인트 수와 프롬프트 수로만 제한된다.

**Temperature Sampling**을 추가하면:
- 같은 모델·프롬프트에서 매번 다른 후보 생성 가능
- 확률 분포의 저확률 영역도 탐색 → 더 창의적/다양한 표현 포함
- 노이즈 없이 오염 리스크 없는 순수 다양성 확보

### 5.2 Sampling 파라미터

```yaml
sampling:
  num_samples: 3          # 체크포인트 × 프롬프트 조합당 샘플 수
  temperature: 0.7        # 낮을수록 greedy에 가깝고, 높을수록 다양
  top_p: 0.9              # nucleus sampling
  include_greedy: true    # greedy 결과도 후보에 포함
```

**온도 가이드라인:**

| temperature | 특성 | 적합한 용도 |
|---|---|---|
| 0.0 | Greedy (결정적) | 기본 출력, reference용 |
| 0.3~0.5 | 약한 sampling | 표현 다양화, 품질 유지 |
| **0.7** | **중간 sampling** | **권장: 다양성과 품질 균형** |
| 1.0 | 원본 분포 | 탐색적 실험 |
| 1.2+ | 과도한 다양성 | 노이즈 과다, 비권장 |

### 5.3 후보 수 예시

| 구성 | 후보 수 | 예상 시간 | 비고 |
|---|---|---|---|
| 3 ckpt × 1 prompt × 3 samples | 9 | ~1.1h | Mode 3 단독 최소 |
| 3 ckpt × 3 prompts × 3 samples | 27 | ~3h | Mode 2+3 결합 |
| 3 ckpt × 7 prompts × 3 samples | **63** | **~7.5h** | 최대 구성 |

### 5.4 비대칭 MBR과의 결합 (권장)

Sampling 후보는 품질 분산이 크므로 **비대칭 MBR**과 결합하는 것이 핵심이다.

```
H (hypothesis pool): greedy + sampled 전체  →  다양한 후보 탐색
R (reference pool):  greedy만              →  신뢰할 수 있는 채점 기준
```

이렇게 하면 sampling의 다양성은 취하면서, 저품질 샘플이 채점 기준을 오염시키는 것을 방지한다.
→ 자세한 내용은 [6절](#6-비대칭-mbr-asymmetric-mbr) 참조.

---

## 6. 비대칭 MBR (Asymmetric MBR)

### 6.1 표준 MBR의 오염 문제

표준 MBR에서 모든 후보는 **hypothesis이자 동시에 pseudo-reference**로 작동한다:

$$y^* = \arg\max_{y \in \mathcal{H}} \sum_{r \in \mathcal{H}} U(y, r)$$

이 구조에서 **저품질 후보가 reference 역할**을 하면:
- 저품질 요약이 채점 기준점으로 작용
- 고품질 후보의 MBR 점수가 인위적으로 낮아질 수 있음
- 한국어 대화 요약처럼 출력 공간이 넓은 태스크에서 특히 위험

### 6.2 비대칭 MBR 구조

$$y^* = \arg\max_{y \in \mathcal{H}} \sum_{r \in \mathcal{R}} U(y, r)$$

- $\mathcal{H}$ (hypothesis pool): 다양성 확보를 위해 넓게 구성 — greedy + sampling, 모든 체크포인트
- $\mathcal{R}$ (reference pool): 품질 기준을 위해 좁게 구성 — greedy 출력만, 또는 상위 체크포인트 greedy만

```
H = {greedy_A, greedy_B, greedy_C,               # 3 ckpt greedy
     sample_A1, sample_A2, sample_A3,             # ckpt_A sampling
     sample_B1, sample_B2, sample_B3, ...}        # 모든 샘플 포함

R = {greedy_A, greedy_B, greedy_C}               # greedy만 (신뢰 가능한 기준)

y* = argmax_{y in H} Σ_{r in R} U(y, r)
```

### 6.3 구현 전략

```yaml
mbr:
  utility: rouge_multi
  use_mecab: true
  weighted: true
  asymmetric: true           # 비대칭 MBR 활성화
  reference_policy: greedy   # reference pool 구성 정책
  # reference_policy 옵션:
  #   greedy      : greedy 출력만 R로 사용 (기본 권장)
  #   top_checkpoints: 상위 N개 체크포인트 greedy만 R로 사용
  #   all         : 표준 MBR (H = R, 비대칭 비활성화)
```

### 6.4 적용 시나리오 요약

| 시나리오 | H | R | 적용 |
|---|---|---|---|
| Mode 1/2 (greedy only) | 모든 greedy | 모든 greedy 동일 | 표준 MBR (H=R) |
| Mode 3 sampling 포함 | greedy + sampled | greedy만 | **비대칭 MBR 권장** |
| 미확인 체크포인트 포함 시 | 전체 | 상위 3 ckpt greedy만 | **비대칭 MBR 필수** |

---

## 7. YAML 설정 관리 (conf/)

### 7.1 디렉토리 구조

```
LLM/conf/
├── checkpoints.yaml                # 체크포인트 레지스트리
├── ensemble_mode1_quick.yaml       # Mode 1: 3 ckpt × 1 prompt = 3 후보 (~38min)
├── ensemble_mode2_recommended.yaml # Mode 2: 3 ckpt × 7 prompts = 21 후보 (~2.6h)
├── ensemble_mode2_full.yaml        # Mode 2 + 비대칭 MBR + BERTScore (~3h)
└── ensemble_mode3_sampling.yaml    # Mode 3: sampling 포함 비대칭 MBR (~4h)
```

### 7.2 체크포인트 레지스트리 (`conf/checkpoints.yaml`)

실제 파일 위치: `LLM/conf/checkpoints.yaml`

```yaml
# 앙상블 대상: lb_confirmed: true 인 체크포인트만 사용
# lb_confirmed: false / null 은 conf/checkpoints.yaml에 등록만 하고 yaml에서 제외
```

### 7.3 설정 파일 구성 요약

| 파일 | Mode | 후보 수 | 예상 시간 | MBR 방식 | 용도 |
|---|---|---|---|---|---|
| mode1_quick | 1 | 3 | ~38min | 표준, ROUGE-multi | 빠른 효과 검증 |
| mode2_recommended | 2 | 21 | ~2.6h | 표준, ROUGE-multi | 핵심 제출용 |
| mode2_full | 2 | 21 | ~3h | 표준, BERTScore+ROUGE | 최고 품질 제출용 |
| mode3_sampling | 3 | 30 | ~4h | **비대칭**, ROUGE-multi | Sampling 다양성 실험 |

---

## 8. CLI 명세

### 8.1 기본 명령 구조

```bash
python run_ensemble.py \
    --config <yaml_파일_경로> \
    --test_file <테스트_CSV> \
    --output_file <결과_CSV> \
    [옵션들...]
```

### 8.2 전체 옵션 명세

```
필수 인수:
  --config CONFIG             앙상블 설정 YAML 파일 경로
  --test_file TEST_FILE        테스트 데이터 CSV (fname, dialogue, topic 컬럼)
  --output_file OUTPUT         최종 결과 CSV 저장 경로

체크포인트/프롬프트 재정의 (YAML 우선, 이 옵션으로 덮어쓰기 가능):
  --checkpoints NAME [NAME ...]   사용할 체크포인트 이름 목록
  --prompts NAME [NAME ...]       사용할 프롬프트 이름 목록

MBR 설정 재정의:
  --utility {rouge1,rouge_multi,bertscore,combined}
                              MBR utility 함수 (기본: rouge_multi)
  --asymmetric                비대칭 MBR 활성화 (H≠R)
  --reference_policy {greedy,top_checkpoints,all}
                              비대칭 MBR의 reference pool 구성 정책
  --no_mecab                  MeCab 비활성화
  --no_weight                 체크포인트 가중치 무시

Sampling 설정 (Mode 3):
  --num_samples N             체크포인트당 sampling 횟수 (0이면 greedy only)
  --temperature FLOAT         sampling 온도 (기본: 0.7)
  --top_p FLOAT               nucleus sampling (기본: 0.9)

실행 제어:
  --resume                    기존 체크포인트별 예측 결과 재사용
  --dry_run                   추론 없이 설정 검증만 수행
  --checkpoint_registry PATH  레지스트리 YAML (기본: conf/checkpoints.yaml)

출력 제어:
  --save_all                  모든 (ckpt × prompt) 조합 결과 저장 → --resume 시 활용
  --verbose                   상세 로그 출력
```

### 8.3 사용 예시

**[예시 1] Mode 1 빠른 검증 (~38분)**
```bash
python run_ensemble.py \
    --config conf/ensemble_mode1_quick.yaml \
    --test_file ../data/test.csv \
    --output_file outputs/submission_mode1.csv
```

**[예시 2] Mode 2 권장 제출용 (~2.6시간)**
```bash
python run_ensemble.py \
    --config conf/ensemble_mode2_recommended.yaml \
    --test_file ../data/test.csv \
    --output_file outputs/submission_mode2.csv \
    --save_all
```

**[예시 3] Mode 3 Sampling + 비대칭 MBR (~4시간)**
```bash
python run_ensemble.py \
    --config conf/ensemble_mode3_sampling.yaml \
    --test_file ../data/test.csv \
    --output_file outputs/submission_mode3.csv \
    --save_all
```

**[예시 4] 중단 후 재개**
```bash
# 기존 체크포인트별 예측 결과 재사용 (--save_all로 저장된 것 활용)
python run_ensemble.py \
    --config conf/ensemble_mode2_recommended.yaml \
    --test_file ../data/test.csv \
    --output_file outputs/submission_mode2.csv \
    --resume
```

**[예시 5] Dev set 성능 평가**
```bash
# dev.csv에 summary 컬럼이 있으면 자동으로 ROUGE 계산
python run_ensemble.py \
    --config conf/ensemble_mode2_recommended.yaml \
    --test_file ../data/dev.csv \
    --output_file outputs/dev_eval_mode2.csv \
    --verbose
```

**[예시 6] CLI로 직접 지정 (YAML 없이)**
```bash
python run_ensemble.py \
    --config conf/ensemble_mode2_recommended.yaml \
    --test_file ../data/test.csv \
    --output_file outputs/custom.csv \
    --checkpoints r4b_response_only qwen14b_lora_v1 \
    --prompts base qa_style topic \
    --utility bertscore \
    --asymmetric
```

---

## 9. 성능 향상 가능성 평가

### 9.1 현재 스코어 분석

| 지표 | 값 | 비고 |
|---|---|---|
| 현재 LB 점수 | **52.0083** | r4b + 7 prompts MBR |
| 내부 dev ROUGE-1 | 0.5716 | MBR 8 prompts 기준 |
| KoBART-scale 상한 | ~52 | PORORO 벤치마크 기준 |
| LLM 7B+ 기대치 | 60~63 | SAMSum 영어 기준 Baichuan2-Sum |

### 9.2 전략별 기대 향상

| 전략 | 예상 ROUGE-1 향상 | 달성 가능 점수 | 핵심 근거 |
|---|---|---|---|
| 현재 (단일 ckpt, 7 prompts) | 기준 | 52.0 | 측정값 |
| ROUGE Multi (1+2+L) 전환 | +0.2~0.5 | ~52.5 | 메트릭 편향 감소 |
| **Mode 1** (3 ckpt × 1 prompt) | +0.5~1.5 | ~53.5 | 체크포인트 다양성 |
| **Mode 2** (3 ckpt × 7 prompts) | +1.0~2.5 | ~54.5 | 두 다양성 축 결합 |
| **Mode 3** (sampling + 비대칭) | +0.5~1.5 추가 | ~55.0 | 확률적 탐색 추가 |
| Mode 2 + BERTScore utility | +0.5~1.5 추가 | ~56.0 | 신경망 메트릭 |

> **현실적 기대:** Mode 2 (3 ckpt × 7 prompts, ROUGE Multi) 시 **+1~2.5점** 향상이 가장 현실적.
> 총 목표: 52.0 → **54~55** 범위 진입.

### 9.3 수확 체감 구간

```
후보 수:   4     7    12    21    28    35    56    84
기대 점수: 52.5  52.8  53.2  53.8  54.5  54.8  55.0  55.2
                                   ↑ 권장    ↑ 수확 체감 진입
```

EMNLP 2024 논문 기준, 실용적 최적 구간은 **21~35 후보**. Mode 2 (21 후보)가 해당 범위 하단에 있다.

### 9.4 비대칭 MBR의 추가 가치

표준 MBR 대비 비대칭 MBR의 이점:

- Sampling 후보의 저품질 결과가 reference pool을 오염시키지 않음
- Greedy 결과를 anchor로 유지하면서 sampling의 다양성을 취함
- 미확인 체크포인트를 H에 추가할 때 R 오염 없이 실험 가능

### 9.5 52.0 돌파 후 추가 전략 (참고)

| 전략 | 비용 | 기대 향상 |
|---|---|---|
| Checkpoint weight averaging (같은 run 내 마지막 3~5 ckpt 평균) | 매우 낮음 | +0.3~0.8 |
| Length-Normalized MBR (길이 편향 보정) | 낮음 | +0.2~0.5 |
| 데이터 증강 재훈련 | 높음 (재학습 필요) | +1~3 |
| 7B+ 모델 추가 파인튜닝 | 매우 높음 | +5~10 |

---

## 10. 구현 로드맵

### Phase 1: 기반 구조

```
□ conf/checkpoints.yaml         체크포인트 레지스트리 (3개 확인된 상위 ckpt 마킹)
□ conf/ensemble_mode*.yaml      설정 파일 4종
□ run_ensemble.py               메인 CLI 스크립트
  - YAML 파싱 + checkpoints.yaml 참조 해석
  - 순차 체크포인트 로딩/언로딩 (VRAM 관리)
  - 기존 prompts/mbr_decoding.py 재사용
  - --save_all / --resume 지원
```

### Phase 2: Mode 1 검증 (~1일)

```
□ 3 체크포인트 × 1 대표 프롬프트 실행
□ Dev set ROUGE 측정 및 체크포인트 기여도 분석
□ MBR 선택 빈도 통계 확인
□ 가중치 조정 실험
```

### Phase 3: Mode 2 풀 실행 (~1일)

```
□ 3 체크포인트 × 7 프롬프트 실행 (21 후보)
□ ROUGE Multi utility 적용
□ Dev vs LB 점수 비교 (오버피팅 여부 확인)
□ 제출
```

### Phase 4: Mode 3 + 비대칭 MBR (~1일)

```
□ Temperature sampling 추가 (temperature=0.7, num_samples=3)
□ 비대칭 MBR 구현 (H = greedy + sampled, R = greedy only)
□ Mode 2 대비 Dev ROUGE 비교
□ BERTScore utility 통합 (klue/roberta-base)
□ 최종 제출
```

---

## 11. 주의사항 및 리스크

### 11.1 VRAM 관리 (최우선)

- Qwen3-14B (4bit quantized) ≈ 8~10GB VRAM
- **체크포인트 순차 로딩 필수**: 동시 로딩 불가
- 추론 후 반드시 `del model; gc.collect(); torch.cuda.empty_cache()` 호출
- `--save_all`으로 중간 결과 저장 → 중단 시 `--resume`으로 복구

### 11.2 Sampling 재현성

- Sampling은 확률적이므로 동일 실행 보장 안 됨
- `torch.manual_seed()` 고정 권장 (실험 비교 시)
- 최종 제출 전 seed 고정 여부 결정

### 11.3 메트릭 편향 주의

- ROUGE-1 단독 MBR: ROUGE는 올라가지만 실제 텍스트 품질 향상 제한적
- 리더보드가 ROUGE 기반이면 ROUGE Multi가 합리적
- 리더보드 평가 방식이 불명확하면 BERTScore 병행 실험 권장

### 11.4 재현성 및 아카이빙

- YAML 설정 파일과 함께 결과 아카이빙
- 어떤 (ckpt × prompt) 조합이 최종 선택되었는지 통계 기록
  (`apply_mbr_to_dataset()`의 선택 빈도 출력 활용)
- 제출 전 `dry_run`으로 설정 검증

---

## 참고 문헌

| 논문 | 핵심 내용 |
|---|---|
| *Improving MBR Decoding with Multi-Prompt* (EMNLP 2024) | 멀티 프롬프트 MBR 유효성 실증, 64 프롬프트까지 이득 |
| *Theoretical Aspects of Bias and Diversity in MBR* (arXiv:2410.15021, 2024) | 다양성이 MBR 성능 천장 결정, bias-diversity decomposition |
| *CUED at ProbSum 2023: Hierarchical Ensemble* (ACL BioNLP 2023) | 체크포인트 앙상블 + MBR으로 요약 공유 태스크 1위 |
| *High Quality Rather than High Model Probability* (TACL 2022) | 신경망 메트릭(BERTScore 등)이 ROUGE보다 MBR utility로 우수 |
| *Mitigating Metric Bias in MBR* (WMT 2024) | 단일 메트릭 MBR의 reward hacking 경고, 복합 utility 권장 |
| *Model-Based MBR for Text Generation* (arXiv:2311.05263, 2023) | BERTScore utility, 길이 정규화 MBR |
| *M-Ped: Multi-Prompt Ensemble Decoding* (arXiv:2412.18299, 2024) | Token-level 멀티 프롬프트 앙상블 (+1.5 BLEU MT) |
