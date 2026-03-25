# 점수 개선 단계별 계획

> KoBART v2 + TTA (현재 **48+**) / Qwen3-14B QLoRA + MBR (현재 **51+**) 기준
> 원칙: **dev 검증 먼저 → 변경 최소 → 효과 큰 것부터**

---

## 전제 파악

| 항목 | KoBART 트랙 | Qwen 트랙 |
|------|-------------|-----------|
| 학습 | `python src/train.py model=kobart_v2` | `LLM/response_only_SFT/` 노트북 |
| 추론 | `inference=tta` (발화 역전 2-way + MBR) | `LLM/` run_prompts × 8종 + MBR |
| 앙상블 | `src/ensemble_cli.py merge` | 동일 |
| dev 평가 | `scripts/evaluate_on_dev.py` | 동일 |
| 현재 상한 | 48+ (LB) | 51+ (LB) |

**중요 점검 사항 — 검증 완료**

| 항목 | 결과 |
|------|------|
| `apply_tta` n_ways>2 실질 변형 | ✅ 확인됨 (2종만 생성). 3-way 실험 후 **원복 결정** (아래 참조) |
| `MBRDecoder` rouge-L vs combined | ✅ **2-way에서는 차이 없음** (499샘플 모두 동일 선택). 3-way에서만 차이 발생 |
| `data.use_topic` 학습·추론 일치 | ✅ 둘 다 `getattr(data_cfg, "use_topic", False)`로 동일 경로 참조. 문제 없음 |
| Qwen3 `<think>` 태그 제거 | ✅ `postprocess.py:36` `re.DOTALL`로 정상 제거, 추론 파이프라인에 연결됨 |

---

## 공통 원칙

1. **dev / OOF 먼저**: LB 제출 전 `scripts/evaluate_on_dev.py` 또는 K-fold OOF로 비교
2. **한 번에 하나**: 추론 파라미터 → 데이터·학습 → 모델 구조 순
3. **Hydra override 우선**: 새 스크립트 작성보다 `-c` / `++` override로 처리

---

## 실험 결과 로그

### KoBART dev Combined ROUGE (260324_run_003/epoch06_0.7962 기준)

| 방법 | R1 | R2 | RL | Combined | 비고 |
|------|-----|-----|-----|---------|------|
| beam4 | 0.1846 | 0.0658 | 0.1786 | **0.4289** | ✅ 단독 최고 (TTA 없이) |
| beam8 | 0.1845 | 0.0649 | 0.1781 | 0.4275 | ❌ beam4보다 -0.0014 |
| 샘플링 MBR (n=10) | 0.1305 | 0.0334 | 0.1260 | 0.2900 | ❌ 대폭 하락 (샘플링 분산 큼) |
| TTA 2-way + rouge-L MBR | 0.1846 | 0.0656 | 0.1786 | **0.4288** | ✅ 현재 최적 (beam4와 동률) |
| TTA 2-way + combined MBR | — | — | — | 0.4288 | rouge-L과 동일 (2-way 순위 역전 없음) |
| TTA 3-way (짝수인덱스역전) + combined MBR | — | — | — | 0.4069 | ❌ -0.022 하락 |
| TTA 3-way (홀수인덱스역전) + rouge-L MBR | — | — | — | 0.4160 | ❌ -0.013 하락 |
| TTA 3-way (랜덤셔플) + rouge-L MBR | — | — | — | 0.4091 | ❌ -0.020 하락 |
| TTA 3-way (랜덤셔플) + combined MBR | — | — | — | 0.4123 | ❌ -0.017 하락 |

**결론 — 단계 2 최종**:
- `beam4 ≈ TTA 2-way` (0.4289 vs 0.4288). TTA는 2× 추론 비용 대비 이득이 미미.
- `beam8` 은 beam4보다 낮음. 빔 확대 무의미.
- **샘플링 MBR(n=10)** 은 크게 하락(0.2900). KoBART 소규모에서 샘플링 다양성 과잉 → beam4 유지.
- 발화 역전 3-way 모두 2-way보다 낮음. `apply_tta` 2-way 원복.
- **현재 최적 설정**: `inference=tta` (beam4 + 2-way TTA + rouge-L MBR), Combined=0.4288~0.4289.

---

## 단계 1 — 베이스라인 고정 (공통, ~1시간)

> 이후 모든 실험의 기준점을 만든다.

```bash
# KoBART: 현재 최고 체크포인트로 dev ROUGE 기록
python scripts/evaluate_on_dev.py inference=tta \
  inference.ckt_path=checkpoints/<best_run_id>/checkpoint-best

# Qwen: 현재 MBR 결과 CSV로 dev ROUGE 기록
python scripts/evaluate_on_dev.py --pred_csv prediction/<qwen_best>.csv
```

- 결과를 `docs/baseline_scores.md` 또는 WandB에 고정 기록
- 이후 모든 실험은 이 수치와 **직접 비교**

---

## 단계 2 — KoBART: 추론만으로 점수 올리기 (재학습 없음)

### 2-A. 디코딩 파라미터 그리드 (예상 효과: +0.3~1.0)

`conf/inference/beam4.yaml` 복사 후 override로 비교:

```bash
# beam 8, length_penalty 조합
python src/inference.py inference=beam8 \
  inference.length_penalty=1.2 \
  inference.ckt_path=checkpoints/<best>/checkpoint-best

# max_length_ratio 조정 (beam4 기준)
python src/inference.py inference=tta \
  inference.max_length_ratio=0.6   # 현재값 확인 후 ±0.1 탐색
```

비교 우선순위: `beam4+TTA(현재)` → `beam8+TTA` → `beam4+sampling MBR`

### 2-B. TTA MBR vs. 샘플링 MBR 비교 — ✅ 완료

결과: beam4(0.4289) ≈ TTA 2-way(0.4288) > beam8(0.4275) >> 샘플링 MBR n=10(0.2900)

**샘플링 MBR은 KoBART에서 역효과**. beam4 단독 혹은 TTA 2-way가 최적.
TTA 오버헤드(2× 추론)는 점수 이득이 없으므로, 빠른 실험 시 beam4 단독 사용 가능.

### 2-C. TTA 3-way 확장 — ❌ 기각 (실험 완료)

짝수인덱스역전 / 홀수인덱스역전 / 랜덤셔플 3가지 변형 모두 dev에서 2-way(0.4288)보다 낮음.
발화 순서 변형 방식으로는 3-way 이득 없음. `apply_tta` 2-way 유지.

---

## 단계 3 — KoBART: 5-fold OOF 앙상블 (재학습 필요, 예상 효과: +0.5~1.5)

```bash
# GroupKFoldTrainer 사용 (src/ensemble.py 기존 구현)
python src/train.py training=full \
  ++training.use_kfold=true \
  ++training.n_splits=5

# 각 fold 예측 → merge
python src/ensemble_cli.py merge \
  prediction/fold0.csv prediction/fold1.csv \
  prediction/fold2.csv prediction/fold3.csv prediction/fold4.csv \
  --output prediction/kobart_kfold_ensemble.csv
```

> 학습 코드 변경 없음. `GroupKFoldTrainer`가 이미 `src/ensemble.py`에 구현됨.

---

## 단계 4 — Qwen: 추론·프롬프트 최적화 (재학습 없음)

### 4-A. 생성 파라미터 소규모 그리드 — ✅ 완료

결과 (dev 100샘플, `mbr_ensemble/r4b_response_only_ckpt`, single "base" prompt):

| 설정 | R1 | R2 | RL | Combined |
|------|-----|-----|-----|---------|
| **greedy max128** | 0.3128 | 0.1347 | 0.3023 | **0.7497** |
| **greedy max256** | 0.3128 | 0.1347 | 0.3023 | **0.7497** |
| sampling t=0.7 p=0.95 max256 | 0.2582 | 0.0908 | 0.2479 | 0.5968 |
| sampling t=0.6 p=0.90 max256 | 0.2868 | 0.1039 | 0.2740 | 0.6647 |

**결론**: `max128 = max256` (truncation 없음). **샘플링은 크게 하락** (-0.08~-0.15).
현재 설정(`greedy, max_new_tokens=128`)이 이미 최적. 변경 불필요.

### 4-B. 프롬프트 수 최적화 (예상 효과: +0.2~0.5)

```bash
# dev에서 8종 각각 ROUGE 측정 후 하위 2~3종 제거
# 축소된 5~6종으로 MBR 재실행 → 노이즈 감소 여부 확인
```

> 변경: 프롬프트 목록 파일(`LLM/prompts/base_prompts.py`)에서 주석 처리만

### 4-C. 후처리 점검 (예상 효과: 버그 픽스 시 +1.0 이상)

```bash
# LLM/prompts/postprocess.py 확인
# Qwen3 thinking 태그 (<think>...</think>) 제거 여부
# 불필요한 줄바꿈·prefix 잔존 여부
```

---

## 단계 5 — Qwen: QLoRA 소규모 스윕 (학습 필요, 예상 효과: +0.5~2.0)

> 단계 4 완료 후 진행. dev 기준으로 검증된 설정만 학습.

| 변수 | 현재 추정 | 탐색 범위 |
|------|-----------|-----------|
| `lora_r` | 16~32 | 32, 64 |
| `lora_alpha` | 32~64 | r × 2 고정 |
| `learning_rate` | 2e-4 | 1e-4, 3e-4 |
| `max_seq_length` | 1024? | 1536 or 2048 (1회) |
| `num_epochs` | 3 | dev loss 기준 조기 종료 |

```bash
# LLM/response_only_SFT/ 노트북에서 파라미터만 변경
# 한 번에 1~2 조합만 → dev ROUGE 비교
```

**과적합 경고**: 51+ 베이스에서 epoch 늘리면 오히려 MBR 이득이 상쇄됨. dev 곡선 필수 확인.

---

## 단계 6 — 교차 앙상블: KoBART + Qwen (예상 효과: +1.0~3.0)

> 두 모델의 오류 패턴이 다르면 앙상블 이득이 크다.

```bash
# 가중치 스윕 (dev ROUGE 기준)
python src/ensemble_cli.py merge \
  prediction/kobart_best.csv \
  prediction/qwen_best.csv \
  --weights 0.3 0.7 \
  --output prediction/ensemble_0.3_0.7.csv

# dev에서 0.2~0.5 범위로 5~7종 후보 생성 → 최고 dev 점수 가중치만 LB 제출
```

> **LB 제출은 2~3종으로 제한** (shake-up 리스크 최소화)

---

## 단계 7 — (선택) 고비용 확장

이 단계는 위 단계들의 이득이 포화됐을 때만 진행한다.

| 항목 | 예상 효과 | 비고 |
|------|-----------|------|
| SimPO / DPO | +1~3 | `LLM/simpo/` 노트북 이미 존재, SFT 안정 후 |
| 더 큰 모델 (Qwen3-32B 등) | 불명확 | 문서상 32B < 14B 사례 존재 |
| KoBART → pko-T5-large 교체 | +0~1 | `conf/model/pko_t5.yaml` 기존 설정 존재 |
| Solar API 정제 (`inference_solar_refine.py`) | +0.5~1.5 | API 비용 고려 |

---

## 실행 요약

```
단계 1: 베이스라인 고정          (1h)   → dev 수치 확정
단계 2: KoBART 추론 파라미터     (2h)   → beam/penalty/TTA vs MBR
단계 3: KoBART 5-fold 앙상블    (4~8h) → 재학습 5회
단계 4: Qwen 추론·프롬프트       (2h)   → 파라미터·후처리
단계 5: Qwen QLoRA 스윕         (4~8h) → 2~3 조합
단계 6: 교차 앙상블             (1h)   → 가중치 탐색 후 LB 제출
단계 7: (선택) 고비용 확장
```

---

## 주요 파일 참조

| 목적 | 파일 |
|------|------|
| Seq2Seq 추론 진입점 | `src/inference.py` |
| TTA 변형 생성 | `src/data/preprocess.py::apply_tta` |
| MBR 후보 선택 | `src/ensemble.py::MBRDecoder` |
| K-fold 학습 | `src/ensemble.py::GroupKFoldTrainer` |
| CSV 앙상블 | `src/ensemble_cli.py merge` |
| dev 평가 | `scripts/evaluate_on_dev.py` |
| Qwen 프롬프트 | `LLM/prompts/base_prompts.py` |
| Qwen 후처리 | `LLM/prompts/postprocess.py` |
| KoBART beam 설정 | `conf/inference/beam4.yaml`, `beam8.yaml` |
| KoBART TTA 설정 | `conf/inference/tta.yaml` |
| KoBART MBR 설정 | `conf/inference/mbr.yaml` |
| Qwen QLoRA 학습 | `LLM/response_only_SFT/대화요약_Qwen3_14B_LoRA_SFT.ipynb` |
| 전체 운영 가이드 | `docs/OPERATION.md` |
| LoRA 연구 가이드 | `docs/RESEARCH_LoRA.md` |
