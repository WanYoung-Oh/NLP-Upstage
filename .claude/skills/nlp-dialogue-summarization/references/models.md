# 모델 선택 가이드

대화 요약 태스크에 적합한 모델 목록과 특징.

---

## 추천 모델 (한국어)

### Encoder-Decoder (Seq2Seq) — 가장 권장

| 모델 | HuggingFace ID | 특징 |
|------|---------------|------|
| **KoBART** (베이스라인) | `digit82/kobart-summarization` | 한국어 요약 전용 fine-tune, 안정적 |
| KoBART base | `gogamza/kobart-base-v2` | 사전학습만 된 KoBART, 직접 fine-tune |
| PEGASUS-ko | `snunlp/KR-ELECTRA-discriminator` | 요약 특화 사전학습 |
| mBART | `facebook/mbart-large-cc25` | 다국어, 한국어 포함 |
| KoT5 | `paust/pko-t5-large` | T5 기반 한국어 모델 |

### Decoder-Only (CLM 기반 요약)

| 모델 | HuggingFace ID | 특징 |
|------|---------------|------|
| KoGPT2 | `skt/kogpt2-base-v2` | 프롬프트 기반 요약 가능 |
| Llama-3-Korean | `beomi/Llama-3-Open-Ko-8B` | 대형 모델, GPU 메모리 주의 |

---

## 모델 교체 방법

`config.yaml`의 `general.model_name`만 수정:

```yaml
general:
  model_name: "gogamza/kobart-base-v2"  # 여기만 변경
```

모델에 따라 `tokenizer.special_tokens` 호환 여부 확인 필요.

---

## 모델 아키텍처별 학습 설정 차이

### BART 계열
```python
from transformers import BartForConditionalGeneration, BartConfig
model = BartForConditionalGeneration.from_pretrained(model_name)
# Seq2SeqTrainer 사용
```

### T5 계열
```python
from transformers import T5ForConditionalGeneration, T5Config
model = T5ForConditionalGeneration.from_pretrained(model_name)
# 입력 앞에 "summarize: " prefix 추가 필요
encoder_input = "summarize: " + dialogue
```

### mBART 계열
```python
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
tokenizer = MBart50TokenizerFast.from_pretrained(model_name, src_lang="ko_KR", tgt_lang="ko_KR")
```

---

## 성능 향상 전략

### 1. 모델 크기 업스케일
- base → large → xl 순으로 파라미터 수 증가
- GPU 메모리 부족 시 `gradient_accumulation_steps` 증가로 보완

### 2. Beam Search 튜닝
```yaml
inference:
  num_beams: 4        # 기본값, 높일수록 품질↑ 속도↓
  no_repeat_ngram_size: 2  # 반복 방지
  length_penalty: 1.0      # >1이면 긴 요약 선호
```

### 3. 학습률 스케줄러
```yaml
training:
  lr_scheduler_type: "cosine"     # cosine, linear, polynomial
  warmup_ratio: 0.1               # 전체 스텝의 10% warmup
```

### 4. 특수 토큰 확장
대화 데이터 특성상 개인정보 마스킹 토큰 추가 필수:
```python
special_tokens = [
    '#Person1#', '#Person2#', '#Person3#', '#Person4#', '#Person5#', '#Person6#', '#Person7#',
    '#PhoneNumber#', '#Address#', '#PassportNumber#', '#SSN#', 
    '#CardNumber#', '#CarNumber#', '#Email#', '#DateOfBirth#'
]
```

### 5. 앙상블
```python
# 여러 체크포인트 예측 결과를 단순 선택 또는 스코어 기반 병합
# 가장 ROUGE 높은 요약문 선택하는 방식 권장
```
