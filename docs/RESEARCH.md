# 일상 대화 요약 경진대회 - 우승 전략 리서치

> **대회**: DialogSum: A Real-life Scenario Dialogue Summarization (Upstage AI Stages #415)
> **목표**: 베이스라인 ROUGE-F1 47.12 이상, 상위권 달성
> **작성일**: 2026-03-12

---

## 1. 대회 개요

### 태스크
- 최소 2명 ~ 최대 7명이 등장하는 일상 multi-turn 대화를 한국어로 요약
- 데이터셋: DialogSum 기반 한국어 번역/구축

### 데이터 규모

| 구분 | 수량 | 비고 |
|------|------|------|
| Train | 12,457 | dialogue + summary 1개 |
| Dev | 499 | dialogue + summary 1개 |
| Test Public | 250 | dialogue + **summary 3개** |
| Test Private | 249 | dialogue + **summary 3개** |

### 핵심 특징
- **공식적 문체**: 채팅과 달리 약어/이모티콘 없음, formal style
- **주요 주제**: 일상 대화(17.5%), 쇼핑(13.9%), 전화 통화(7.3%), 직업 면접(6.8%), 음식 주문(6.3%) 등 30개 이상
- **개인정보 마스킹**: `#PhoneNumber#`, `#Address#`, `#DateOfBirth#`, `#PassportNumber#`, `#SSN#`, `#CardNumber#`, `#CarNumber#`, `#Email#`
- **길이 분포**: Right-skewed, 대화 길이 >> 요약 길이 (요약은 대화의 약 20%)

---

## 2. 평가 방법

### 수식
```
Score = mean(ROUGE-1-F1) + mean(ROUGE-2-F1) + mean(ROUGE-L-F1)
```

### 핵심 사항
- **형태소 기반 토크나이징**: 한국어 특성상 조사/어미 분리 후 점수 산출
  - 예: `호킨스 의사는` → `호킨스 / 의사 / 는`
- **Test의 3개 reference**: dialogue 1개에 summary 3개 존재 → 3개 각각과 개별 채점 후 종합
  - 실제로는 3개 중 어떤 reference와도 overlap이 높아야 유리

### 정답 요약문 작성 기준 (중요!)
1. 대화의 **가장 중요한 정보** 전달
2. **간략하게** (대화 길이의 20% 이내)
3. **명명된 개체 보존** (사람 이름, 기업명 등)
4. **관찰자 관점**으로 작성 (화자의 의도를 이해)
5. 은어/약어 없이 **공식적 언어** 사용

### 베이스라인 성능
- 모델: `digit82/kobart-summarization` (KoBART)
- Public 250개 기준 **ROUGE-F1: 47.1244**

---

## 3. 접근 전략 로드맵

```
Phase 1: 베이스라인 재현 & 빠른 실험
Phase 2: 모델 업그레이드
Phase 3: 데이터/학습 전략 고도화
Phase 4: LLM 활용
Phase 5: 앙상블 & 후처리 최적화
```

---

## 4. Phase 1 — 베이스라인 재현 & 빠른 실험

### 목표: 47 → 50+

### 4.1 하이퍼파라미터 Sweep (즉시 시도 가능)

**Hydra sweep (빠른 탐색)**:
```bash
python src/train.py -m \
  training.learning_rate=1e-5,3e-5,5e-5 \
  training.per_device_train_batch_size=32,50
```

**Optuna 자동 최적화 (50 trial, validation ROUGE 최대화)**:
```python
import optuna
def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    warmup = trial.suggest_float('warmup_ratio', 0.05, 0.2)
    # Trainer 실행 후 dev ROUGE 반환
    return dev_rouge
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```
→ Hydra + Optuna sweeper 연동 가능 (`hydra-optuna-sweeper` 패키지)

| 파라미터 | 베이스라인 | 시도 범위 | 근거 |
|----------|-----------|-----------|------|
| learning_rate | 1e-5 | **2e-5~1e-4** | STRATEGY 기반, 더 넓은 탐색 범위 |
| num_train_epochs | 20 | 5~10 (SOLAR), 20~50 (T5/BART) | 모델 크기에 따라 다름 |
| warmup_ratio | 0.1 | 0.05, 0.1, 0.15 | 작은 데이터셋에서 warmup 효과 |
| weight_decay | 0.01 | 0.01, 0.05, 0.1 | 정규화 강도 조절 |
| num_beams | 4 | 4, 8, 12 | beam 수 증가 → 품질 향상 (속도↓) |
| no_repeat_ngram_size | 2 | 2, 3 | 반복 표현 방지 |
| gradient_accumulation | 1 | 1, 2, 4 | SOLAR QLoRA는 4 권장 |

### 4.2 생성 파라미터 튜닝

```python
# 추가로 시도할 생성 파라미터
generate_model.generate(
    input_ids=...,
    num_beams=8,               # 4 → 8
    no_repeat_ngram_size=3,    # 2 → 3
    length_penalty=1.0,        # 1.0 (중립), >1.0 (긴 요약), <1.0 (짧은 요약)
    min_length=10,             # 너무 짧은 요약 방지
    max_length=100,
    early_stopping=True,
)
```

**length_penalty 튜닝 전략**: test reference가 3개이므로 다양한 길이를 커버하는 요약이 유리. `length_penalty=1.2~1.5` 시도 권장.

---

## 5. Phase 2 — 모델 업그레이드

### 목표: 50 → 56+

### 5.1 한국어 요약 특화 모델 후보

| 모델 | HuggingFace ID | 아키텍처 | 실측/예상 ROUGE | 우선순위 |
|------|----------------|----------|----------------|----------|
| KoBART (베이스라인) | `digit82/kobart-summarization` | BART | **47.12** (실측) | - |
| gogamza KoBART v2 | `gogamza/kobart-base-v2` | BART | **49.48** (실측) | ★★★★ |
| KoT5-summarization | `psyche/KoT5-summarization` | T5 | **49.87** (실측) | ★★★★★ |
| pko-T5-large | `paust/pko-t5-large` | T5 | ~55+ (예상) | ★★★★ |
| pko-T5-base | `paust/pko-t5-base` | T5 | 빠른 실험용 | ★★★ |
| KoT5 (eenzeenee) | `eenzeenee/t5-base-korean-summarization` | T5 | 요약 특화 fine-tune | ★★★ |
| **SOLAR-KO-10.7B** | `upstage/SOLAR-10.7B-Instruct-v1.0` | LLaMA-based | **1등 솔루션** (QLoRA) | ★★★★★ |
| mT5-large | `google/mt5-large` | T5 | ~53+ (예상) | ★★★ |
| mBART-large | `facebook/mbart-large-cc25` | BART | 다국어 BART | ★★ |
| PEGASUS (다국어) | `google/pegasus-large` | PEGASUS | 요약 태스크 전용 pre-train | ★★★ |

### 5.2 모델 선택 가이드

**SOLAR-KO-10.7B (1등 솔루션)**:
- Upstage SOLAR 10.7B 한국어 instruction-tuned 모델
- **QLoRA**로 fine-tuning 필수 (full fine-tuning은 VRAM 부족)
- 컨텍스트 길이 4k → 긴 대화도 처리 가능
- QLoRA 설정: `r=64, alpha=128, 4-bit quantization (BitsAndBytes), gradient_accumulation=4`
- 학습 epoch은 5~10으로 적게, lr=2e-5~1e-4

**KoT5-summarization 추천 이유**:
- `psyche/KoT5-summarization`: 이 대회와 유사한 설정에서 ROUGE **49.87** 실측
- T5-large 기반 한국어 요약 특화 → 바로 활용 가능
- `gogamza/kobart-base-v2`: **49.48** 실측, 가볍고 빠름

**PEGASUS 특별 고려 사유**: PEGASUS는 요약 태스크를 위해 "Gap Sentence Generation(GSG)" 방식으로 사전 학습됨. 요약 성능이 BART 대비 일반적으로 우수하나 한국어 전용 PEGASUS가 없어 multilingual 버전 사용 필요.

**pko-T5-large 추천 이유**:
- PAUST(Kakao) 팀이 한국어 대규모 코퍼스로 사전 학습
- T5 prefix 기반으로 다양한 태스크 적용 용이
- `summarize: {dialogue}` prefix로 요약 유도 가능

**Hydra config 설정 (conf/model/pko_t5.yaml)**:
```yaml
name: pko_t5_large
model_name: "paust/pko-t5-large"
architecture: "t5"
prefix: "summarize: "
```

**Hydra config 설정 (conf/model/solar_qlora.yaml)**:
```yaml
name: solar_qlora
model_name: "upstage/SOLAR-10.7B-Instruct-v1.0"
architecture: "causal_lm"
qlora:
  r: 64
  alpha: 128
  bits: 4
  gradient_accumulation_steps: 4
```

### 5.3 T5 계열 주의사항
- T5는 입력에 task prefix 추가 필요: `"summarize: " + dialogue`
- `BartForConditionalGeneration` → `T5ForConditionalGeneration`으로 교체
- tokenizer `return_token_type_ids=False` 유지

### 5.4 SOLAR QLoRA 주의사항
- `bitsandbytes` 라이브러리 필요: `pip install bitsandbytes`
- `peft` 라이브러리로 LoRA 적용: `pip install peft`
- Causal LM이므로 Seq2Seq와 다른 데이터 포맷 필요 (instruction-following 형식)

---

## 6. Phase 3 — 데이터/학습 전략 고도화

### 목표: 56 → 60+

### 6.0 데이터 클리닝 (베이스라인 +3% 기대)

**텍스트 노이즈 제거**:
```python
import re
def clean_text(text):
    text = re.sub(r'[ㄱ-ㅎㅏ-ㅣ]', '', text)   # 단독 자음/모음 제거
    text = re.sub(r'\([^)]*\)', '', text)       # 괄호 및 내용 제거
    text = re.sub(r'[!?~"]{2,}', '', text)      # 반복 특수기호 제거
    return text.strip()

train_df['dialogue'] = train_df['dialogue'].apply(clean_text)
```

**길이 기반 필터링 (이상치 제거)**:
- dialogue 길이 1500자 초과: 모델 컨텍스트 초과 우려 → 제거 또는 truncation
- summary 길이 50자 미만 / 250자 초과: 품질 의심 → IQR 기반 상위 5% drop

**Topic 추출 (선택)**:
- KoBART classifier로 대화 주제를 분류하여 `dialogue_with_topic` 컬럼 생성
- 입력 포맷: `[주제: 쇼핑] {dialogue}` → 모델이 주제별 요약 스타일 학습 가능
- GroupKFold CV 시 topic 그룹 기반 분할에도 활용 (섹션 8.1 참고)

### 6.1 입력 포맷 개선

**현재 (베이스라인)**:
```
#Person1#: 안녕하세요, 스미스씨. ...
#Person2#: 건강검진을 받는 것이 좋을 것 같아요.
```

**개선 방향 1 - 화자 분리 강조**:
```
[화자A] 안녕하세요, 스미스씨. ...
[화자B] 건강검진을 받는 것이 좋을 것 같아요.
```

**개선 방향 2 - 대화 구조 명시**:
```
대화:
#Person1#: 안녕하세요 ...
#Person2#: 건강검진을 ...
요약:
```
→ decoder 입력에 "요약: " prefix 추가하여 생성 방향 유도

### 6.2 Decoder 입력 전략

| 전략 | decoder_input | 기대 효과 |
|------|---------------|-----------|
| 베이스라인 | `<s> + summary` | 기본 |
| Prefix 유도 | `<s>요약: ` | 공식적 요약 스타일 유도 |
| 키워드 추출 + 생성 | `<s>[키워드] + ` | 핵심 단어 기반 생성 |

### 6.3 데이터 증강 (train 12k → 최대 20k+ 확대 목표)

**방법 1: Train + Dev 합산 학습**
- 최종 submission 직전, train+dev(12,956개) 전체로 재학습
- early stopping 기준: dev ROUGE → 고정 epoch 수 사용

**방법 2: Back-translation (번역 증강, ×2배)**
```
한국어 대화 → 영어 번역 (googletrans / Naver Papago) → 한국어 역번역
→ 의미 보존 + 표현 다양화
→ 약 2배 데이터 확보
```

**방법 3: EDA/AEDA (nlpaug)**
```python
import nlpaug.augmenter.word as naw
aug = naw.SynonymAug(aug_src='ko')
augmented = [aug.augment(text)[0] for text in train_df['dialogue']]
# synonym 교체 15%, 단어 삭제 10%, 삽입 10%
```
- 경량 증강으로 빠르게 데이터 다양성 확보
- dialogue에는 적용하되 summary에는 적용 주의

**방법 4: 외부 데이터 활용**
- **AI Hub "일상 대화 요약" 데이터**: 약 20k 대화, 대회 도메인과 동일
- **SAMSum 한국어 번역**: m2m100 모델로 번역하여 활용 (영어 대화 요약 데이터셋)

**방법 5: LLM 합성 데이터 생성**
- GPT-4o 또는 Solar API로 topic별 대화 합성 (few-shot prompting)
- 데이터 부족 주제 (interview, business 등) 집중 보강
- 단, 합성 데이터는 ROUGE 점수 필터링 후 사용

**방법 6: Solar API pseudo-label 생성**
- Solar API로 train 데이터에 대한 추가 summary 생성
- 기존 1개 → Solar 생성 1개 추가 = 2배 데이터
- 단, 노이즈 필터링 필요 (ROUGE 점수 기반 필터)

### 6.4 특수 토큰 확장

베이스라인에 있는 토큰:
```
#Person1#, #Person2#, #Person3#, #PhoneNumber#, #Address#, #PassportNumber#
```

데이터 내 추가 마스킹 토큰 (PDF EDA 확인):
```
#DateOfBirth#, #SSN#, #CardNumber#, #CarNumber#, #Email#
```
→ **누락된 special token 추가로 tokenization 품질 향상**

### 6.5 Label Smoothing

```python
# Seq2SeqTrainingArguments에 추가
label_smoothing_factor=0.1  # 과적합 방지, 일반화 성능 향상
```

---

## 7. Phase 4 — LLM 활용 전략

### 목표: Solar API 기반 고성능 요약

### 7.1 Solar API (Upstage solar-1-mini-chat)

베이스라인에서 이미 제공. 핵심은 **프롬프트 엔지니어링**.

**System Prompt 최적화**:
```python
system_prompt = """당신은 한국어 대화 요약 전문가입니다.
다음 지침에 따라 대화를 요약하세요:
1. 대화의 가장 중요한 정보만 포함
2. 대화 길이의 20% 이내로 간결하게
3. 사람 이름, 기업명 등 고유명사 보존
4. 관찰자 시점으로 작성 (예: "Person1은 ...한다")
5. 공식적인 언어 사용 (은어/약어 금지)"""
```

**Few-shot 전략**:
```python
# Multi-turn few-shot (베이스라인보다 발전된 형태)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"대화:\n{sample_dialogue1}\n\n요약:"},
    {"role": "assistant", "content": sample_summary1},
    {"role": "user", "content": f"대화:\n{sample_dialogue2}\n\n요약:"},
    {"role": "assistant", "content": sample_summary2},
    {"role": "user", "content": f"대화:\n{dialogue}\n\n요약:"},
]
```

**Few-shot sample 선택 전략**:
- 랜덤 선택 대신 **BM25 또는 임베딩 유사도**로 test와 유사한 train sample 선택
- 대화 주제가 비슷한 few-shot → 더 높은 성능

### 7.2 LLM 생성 파라미터

```python
client.chat.completions.create(
    model="solar-1-mini-chat",
    messages=messages,
    temperature=0.1,   # 낮을수록 안정적/결정적 → 요약에 유리
    top_p=0.9,
    max_tokens=150,
)
```

### 7.3 Solar API vs Fine-tuning 성능 비교 계획

| 접근법 | 예상 ROUGE | 비용 | 재현성 |
|--------|-----------|------|--------|
| KoBART fine-tuning (baseline) | **47.12** (실측) | 낮음 | 높음 |
| KoT5-summarization fine-tuning | **49.87** (실측) | 낮음 | 높음 |
| gogamza/kobart-base-v2 | **49.48** (실측) | 낮음 | 높음 |
| pko-T5-large fine-tuning | ~55+ | 중간 | 높음 |
| SOLAR-KO-10.7B QLoRA | ~60+ (1등 솔루션) | 높음 (GPU) | 높음 |
| Solar API zero-shot | ~45-50 | API 비용 | 중간 |
| Solar API few-shot (3-shot) | ~52-57 | API 비용 | 중간 |
| Solar API + prompt engineering | ~55-60 | API 비용 | 중간 |

---

## 8. Phase 5 — 앙상블 & 후처리

### 목표: 60 → 최상위권

### 8.1 앙상블 전략 (5Fold CV + soft voting → +3% 기대)

**Step 1: GroupKFold CV로 OOF 예측 생성**
```python
from sklearn.model_selection import GroupKFold
# topic 컬럼을 그룹으로 사용 → 같은 주제가 train/val에 분산되도록
gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df['topic'])):
    # 각 fold 학습 후 OOF 예측 저장
    oof_preds[val_idx] = model.predict(val_data)
```
→ LB shake-up 방지 효과, 안정적인 모델 선택 가능

**Step 2: 가중치 앙상블 (soft voting)**
```
SOLAR(0.5) + KoT5(0.3) + KoBART(0.2)
→ 각 모델의 output 텍스트를 logit 수준에서 가중 평균
  (같은 vocab이 아닌 경우 output text 기반 투표)
```
- 가중치는 OOF ROUGE 점수에 비례하여 결정

**Step 3: 체크포인트 앙상블 (같은 모델)**:
```
- epoch 15, 17, 20 체크포인트의 생성 확률 평균
- 또는 각 출력에 대해 ROUGE 기반 최고 선택
```

**앙상블 방법론 비교**:
1. **Output Voting**: N개 모델 요약 중 상호 ROUGE 최고 선택
2. **Probability Averaging**: 토큰 확률 평균 (동일 아키텍처끼리)
3. **Minimum Bayes Risk (MBR) Decoding**: 후보들 중 평균 ROUGE 최고 선택 → **강력 추천**

**MBR Decoding 구현 아이디어**:
```python
# 1개 dialogue에 대해 N개 후보 생성 (temperature sampling)
candidates = [model.generate(..., do_sample=True, temperature=0.7) for _ in range(10)]
# 후보들 사이의 평균 ROUGE가 가장 높은 것 선택
best_idx = argmax([mean(rouge(c, others)) for c in candidates])
```

**TTA (Test-Time Augmentation, 8-way)**:
```python
tta_augs = [
    lambda x: x,                          # 원본
    lambda x: reverse_sentences(x),       # 발화 순서 뒤집기
    lambda x: add_noise(x, sigma=0.1),    # 임베딩 노이즈
    lambda x: crop(x, ratio=0.9),         # 앞 90% crop
    # ... 총 8가지 변형
]
tta_preds = [model.generate(tokenizer(aug(dialogue))) for aug in tta_augs]
final_pred = vote_by_rouge(tta_preds)  # 상호 ROUGE 최고 선택
```

### 8.2 후처리 파이프라인

```python
def postprocess(summary):
    # 1. 특수 토큰 제거 (베이스라인 기존)
    for token in ['<usr>', '<s>', '</s>', '<pad>']:
        summary = summary.replace(token, ' ')

    # 2. 과도한 공백 정리
    summary = re.sub(r'\s+', ' ', summary).strip()

    # 3. 문장 끝 마침표 보장
    if summary and not summary.endswith(('.', '!', '?')):
        summary += '.'

    # 4. 반복 문장 제거
    sentences = summary.split('. ')
    seen = set()
    result = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            result.append(s)
    summary = '. '.join(result)

    return summary
```

### 8.3 ROUGE 최적화 관점 후처리

test의 3개 reference 각각에 대해 채점되므로:
- **너무 짧은 요약 금지**: ROUGE recall 하락
- **너무 긴 요약도 주의**: ROUGE precision 하락
- **핵심 명사/동사 보존**: ROUGE-1, ROUGE-2 향상
- **어순 유지**: ROUGE-L (최장 공통 부분 수열) 향상 → 원문의 시제/어순과 유사하게

---

## 9. ROUGE 점수 분석 및 인사이트

### 9.1 형태소 기반 ROUGE의 특성

```
원문:  호킨스 의사는 매년 건강검진을 받는 것을 권장합니다.
토큰화: 호킨스 / 의사 / 는 / 매년 / 건강 / 검진 / 을 / 받 / 는 / 것 / 을 / 권장 / 합니다 / .
```

**함의**:
- 조사(`는`, `을`, `이`) 처리가 점수에 영향 → 문법적으로 정확한 문장 생성이 중요
- 복합 명사 분리됨 (`건강검진` → `건강/검진`) → 핵심 명사 포함 여부가 ROUGE-1에 크게 기여

### 9.2 3개 Reference의 의미

test에서 동일 대화에 3개의 다른 summary가 존재한다는 것은:
- 여러 annotator가 서로 다른 핵심을 선택했을 가능성
- **모델은 가장 "평균적인" 요약**을 생성하는 것이 유리
- 한 특정 관점에 치우치지 않고 **대화의 전체 핵심**을 포괄해야 함

---

## 10. 실험 우선순위 요약

```
[즉시] Phase 1: 하이퍼파라미터 sweep + 생성 파라미터 최적화
         도구: Hydra sweep + Optuna 50 trial
         예상 점수: 47 → 50~52

[1주차] Phase 2: 모델 교체 실험
         우선순위: KoT5-summarization(49.87 실측) → pko-T5-large → SOLAR QLoRA
         예상 점수: 52 → 55~60

[1~2주차] Phase 3: 데이터 고도화
         - 텍스트 클리닝 + 길이 필터 (baseline +3% 기대)
         - 누락 special token 추가, label smoothing
         - 외부 데이터(AI Hub 20k) + Back-translation
         - train+dev 합산 학습 (최종 제출 직전)
         예상 점수: 55 → 57~62

[2주차] Phase 4: Solar API few-shot + 프롬프트 엔지니어링
         예상 점수: 독립적으로 ~55~60

[2~3주차] Phase 5: 5Fold CV + 앙상블 + TTA
         - SOLAR(0.5) + KoT5(0.3) + KoBART(0.2) 가중치 앙상블
         - MBR Decoding, TTA 8-way
         예상 점수: 60+
```

---

## 11. 실험 관리 체크리스트

### 매 실험마다 기록할 항목
- [ ] 모델명 & 체크포인트
- [ ] 주요 하이퍼파라미터 (lr, batch, epoch, beams)
- [ ] Dev ROUGE-1 / ROUGE-2 / ROUGE-L / Total
- [ ] Public Test ROUGE-F1
- [ ] 특이사항 (수렴 속도, 오류 등)

### WandB 활용
```bash
# run name을 의미있게 설정
python src/train.py \
  wandb.name="pko-t5-large_lr3e-5_epoch30_beam8"
```

### 재현성 확보
- `seed: 42` 고정 (이미 베이스라인 설정)
- Hydra로 실험 config 자동 저장 (`outputs/` 디렉토리)

---

## 12. 참고 자료

### 데이터셋 원본
- DialogSum (영어): Chen et al., 2021 - `https://arxiv.org/abs/2105.06762`

### 관련 연구
- PEGASUS (요약 특화 사전학습): Zhang et al., 2020
- BART (seq2seq 언어 모델): Lewis et al., 2019
- MBR Decoding for NLG: Müller & Sennrich, 2021
- Korean NLP benchmarks: KLUE 논문 (Park et al., 2021)

### HuggingFace 모델 링크
- `digit82/kobart-summarization`: KoBART 요약 특화 (baseline)
- `gogamza/kobart-base-v2`: KoBART v2, ROUGE 49.48 실측
- `psyche/KoT5-summarization`: KoT5 요약 특화, ROUGE 49.87 실측
- `paust/pko-t5-large`: 한국어 pko-T5 Large
- `eenzeenee/t5-base-korean-summarization`: T5 한국어 요약
- `upstage/SOLAR-10.7B-Instruct-v1.0`: SOLAR 10.7B (1등 솔루션, QLoRA 필요)

### 추가 라이브러리
- `bitsandbytes`: SOLAR QLoRA 4-bit 양자화
- `peft`: LoRA 적용
- `nlpaug`: EDA/AEDA 데이터 증강
- `optuna` + `hydra-optuna-sweeper`: 자동 하이퍼파라미터 최적화
- `konlpy` (Okt): 형태소 기반 ROUGE 평가

### 대회 제공 베이스라인 성능
- KoBART fine-tuning: **ROUGE-F1 47.1244** (public 250개 기준)
- 학습 시간: 약 21분 (Stages GPU), 추론 시간: 약 13초
