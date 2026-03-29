# 데이터 증강 계획 (Data Augmentation via Solar API)

## 목표

- 원본: `/data/ephemeral/home/NLP/data/train.csv` (12,457행)
- 증강 결과: 약 3,000개 신규 샘플 생성 → `train_augmented.csv`로 저장
- 생성 방법: 대화(dialogue)의 앞 50% 또는 뒷 50%를 삭제 후, Solar API로 삭제된 부분을 재생성
- 생성 조건: 원본 `summary`와 `topic`에 일치하는 내용으로 생성

---

## 1. 데이터 구조

```
fname     : 샘플 ID (train_0, train_1, ...)
dialogue  : 멀티턴 대화 (#Person1#:, #Person2#: 포맷)
summary   : 한 문장 요약
topic     : 대화 주제 (예: 건강검진, 취업 면접 등)
```

- 평균 턴 수: 9.5턴 (min: 2, max: 59)
- 총 고유 topic: 9,235개

---

## 2. 증강 방식

### 2-1. 샘플 선택
- 원본 12,457개 중 **3,000개 무작위 샘플** 추출 (seed=42)
- 단, 최소 턴 수 ≥ 4인 샘플만 대상 (절반 삭제 시 최소 2턴 이상 보존)

### 2-2. 삭제 방향 결정 (50:50 랜덤)
| 방향 | 보존 부분 | 생성 대상 |
|------|----------|----------|
| 앞 50% 삭제 | 뒤쪽 절반 턴 | 앞쪽 절반 턴 (도입부) |
| 뒤 50% 삭제 | 앞쪽 절반 턴 | 뒤쪽 절반 턴 (마무리) |

- 턴 분할: `n_keep = n_turns // 2` (정수 나눗셈)
- 생성해야 할 턴 수 = 원본 턴 수 - n_keep

### 2-3. Solar API 호출 방식

**모델**: `solar-pro` (기본값, 품질 우선) 또는 `solar-1-mini-chat`

**시스템 프롬프트**:
```
당신은 한국어 대화 생성 전문가입니다.
주어진 조건에 맞게 자연스러운 한국어 대화를 생성하세요.
반드시 #Person1#:, #Person2#: 형식을 사용하세요.
```

**유저 프롬프트 (뒤 50% 삭제 → 앞부분 생성 불필요, 뒷부분 생성)**:

> 뒷부분 삭제 예시:
```
주제: {topic}
요약: {summary}

아래는 대화의 앞부분입니다:
{existing_dialogue}

위 대화에 이어지는 자연스러운 뒷부분 대화를 {n_gen}턴 생성하세요.
- #Person1#:, #Person2#: 형식을 반드시 사용하세요
- 주제와 요약 내용에 부합해야 합니다
- 기존 대화 흐름과 자연스럽게 연결되어야 합니다
- 대화만 출력하고 다른 설명은 쓰지 마세요
```

> 앞부분 삭제 예시:
```
주제: {topic}
요약: {summary}

아래는 대화의 뒷부분입니다:
{existing_dialogue}

위 대화 앞에 오는 자연스러운 도입부 대화를 {n_gen}턴 생성하세요.
- #Person1#:, #Person2#: 형식을 반드시 사용하세요
- 주제와 요약 내용에 부합해야 합니다
- 뒷부분 대화와 자연스럽게 연결되어야 합니다
- 대화만 출력하고 다른 설명은 쓰지 마세요
```

### 2-4. 최종 dialogue 구성
- **뒷부분 생성**: `기존 앞부분 + 생성된 뒷부분`
- **앞부분 생성**: `생성된 앞부분 + 기존 뒷부분`

---

## 3. 출력 형식

```
fname     : aug_0000, aug_0001, ... (증강 샘플 구분)
dialogue  : 재구성된 전체 대화
summary   : 원본 그대로 유지
topic     : 원본 그대로 유지
```

저장 경로: `/data/ephemeral/home/NLP/data_aug/train_augmented.csv`

---

## 4. 구현 파일

| 파일 | 역할 |
|------|------|
| `data_aug/generate_aug.py` | 메인 증강 스크립트 |
| `data_aug/prompt_templates.py` | 프롬프트 템플릿 모음 |

---

## 5. 구현 상세

### generate_aug.py 구조

```python
# 주요 처리 흐름
1. load_dotenv('/data/ephemeral/home/NLP/.env')  → UPSTAGE_API_KEY 로드
2. train.csv 로드 → turns ≥ 4 필터링
3. 3,000개 무작위 샘플링 (seed=42)
4. 각 샘플:
   a. dialogue를 턴 단위로 파싱 (regex: #Person\d+#:)
   b. 앞/뒤 삭제 방향 랜덤 결정
   c. 보존 부분 / 생성 타깃 분리
   d. Solar API 호출 (retry 3회, rate limit 대응)
   e. 응답 파싱 → #PersonN#: 형식 검증
   f. 최종 dialogue 조합
5. 결과를 CSV에 점진적 저장 (--resume 지원)
```

### 에러 처리
- API 타임아웃: 30초 후 retry
- Rate limit (429): exponential backoff (1s → 2s → 4s)
- 생성 실패 (형식 불일치): 해당 샘플 스킵, 로그 기록
- 중간 저장: 100개마다 체크포인트 (`aug_checkpoint.csv`)

### Rate Limit 예상
- Solar API 무료 티어: 분당 약 60 req
- 3,000개 생성 예상 시간: **약 60~90분**

---

## 6. 실행 방법

```bash
cd /data/ephemeral/home/NLP
python data_aug/generate_aug.py \
    --input   data/train.csv \
    --output  data_aug/train_augmented.csv \
    --n_aug   3000 \
    --seed    42 \
    --model   solar-pro \
    --resume              # 중단 후 이어서 실행 시
```

### 옵션 설명

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--input` | `data/train.csv` | 원본 데이터 경로 |
| `--output` | `data_aug/train_augmented.csv` | 결과 저장 경로 |
| `--n_aug` | 3000 | 생성할 증강 샘플 수 |
| `--seed` | 42 | 랜덤 시드 |
| `--model` | `solar-pro` | Solar 모델명 |
| `--resume` | False | 체크포인트에서 재개 |

---

## 7. 품질 검증

생성 후 아래 기준으로 필터링:

1. **형식 검증**: `#Person1#:`, `#Person2#:` 패턴 포함 여부
2. **길이 검증**: 생성된 턴 수 ≥ 원본 타깃 턴 수 × 0.5 (너무 짧은 생성 제거)
3. **중복 제거**: 원본 dialogue와 동일한 샘플 제거

---

## 8. 향후 활용

- 증강 데이터를 원본 train.csv와 병합하여 SFT 재학습에 활용
- 증강 비율 조정 (1:1, 1:2 등) 실험으로 최적 비율 탐색
