Upstage AI Lab 대화 요약 대회에서 ROUGE 50+를 위한 최적 전략입니다. 1등 솔루션(SOLAR-QLoRA 앙상블)을 기반으로 사용자의 CV/앙상블 경험을 반영했습니다.

## 1. 데이터 전처리

대화 길이 불균형/노이즈 제거로 baseline +3%.

- 자음/모음/특수기호(소괄호, 반복) 제거: 정규표현식.
- 화자/개인정보 토큰화: #Person1#, #Phone# 등.
- 토크나이저 전처리: 길이 필터 (dialogue <1500, summary 50-250), IQR 이상치 drop (상위 5%).
- Topic 추출: KoBART classifier로 dialogue_with_topic 생성.

```python
import re
from konlpy.tag import Okt
def clean_text(text):
    text = re.sub(r'[ㄱ-ㅎㅏ-ㅣ!?~".]', '', text)  # 자음/모음 제거
    text = re.sub(r'\([^)]*\)', '', text)  # 괄호
    return text
train_df['dialogue'] = train_df['dialogue'].apply(clean_text)
```

## 2. 데이터 증강

train 5k → 20k 확대, AI Hub 활용으로 +5%.

- **Back-translation**: ko→en→ko (googletrans 또는 Naver Papago, 2배).
- **EDA/AEDA**: nlpaug 라이브러리 (synonym 15%, delete 10%, insert 10%).
- **외부 데이터**: AI Hub "일상 대화" (20k) + SAMSum 한국어 번역 (m2m100).
- **합성**: GPT-4o prompt로 topic별 대화 생성 (few-shot).

```python
import nlpaug.augmenter.word as naw
aug = naw.SynonymAug(aug_src='ko')
augmented_dialogues = [aug.augment(text)[0] for text in train_df['dialogue']]
```

## 3. 모델 선택

SOTA 3종: SOLAR (1등), KoT5 (안정), KoBART-v2 (baseline).

- **SOLAR-KO-10.7B-instruct**: QLoRA 필수, 컨텍스트 4k.
- **psyche/KoT5-summarization**: T5-large 한국어 특화, ROUGE 49.87.
- **gogamza/kobart-base-v2**: 49.48 기록 모델, 가벼움. [huggingface](https://huggingface.co/psyche/KoT5-summarization)

## 4. Hyperparameter 최적화

Optuna로 50trial, validation ROUGE 최대화.

- **공통**: lr=2e-5~1e-4, cosine scheduler, warmup_ratio=0.1, epoch=5-10, batch=8-16.
- **SOLAR QLoRA**: r=64, alpha=128, 4bit (BitsAndBytes), gradient_accum=4.
- **평가**: predict_with_generate (num_beams=4), rouge-korean (Okt tokenizer).

```python
import optuna
def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    # trainer 실행 후 rouge 반환
```

WandB sweep으로 병렬 튜닝.

## 5. 앙상블 + TTA

5Fold + soft voting으로 +3%, LB shake-up 방지. [perplexity](https://www.perplexity.ai/search/0ba7300f-2fef-4c5d-a31a-4dbb900c83b1)

- **CV**: GroupKFold(n_splits=5, topic 그룹), OOF 저장.
- **앙상블**: SOLAR(0.5) + KoT5(0.3) + KoBART(0.2), weighted average logits.
- **TTA**: 8-way (flip 순서, noise σ=0.1, crop 90%), 생성 평균.

```python
# TTA 예시
tta_preds = []
for aug in tta_augs:
    input_aug = aug(test_dialogue)
    pred = model.generate(tokenizer(input_aug))
    tta_preds.append(pred)
final_pred = np.mean(tta_preds, axis=0)
```
