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

### KoBART dev Combined ROUGE (260324_run_003/epoch06_0.7962 기준

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

### 2-A. 디코딩 파라미터 그리드 — ✅ 완료

결과 (dev 499샘플, `260324_run_003/epoch06_0.7962`):

| 설정 | Combined | vs 기준 |
|------|---------|---------|
| **beam4 mnt=100 lp=1.0** (현재) | **0.4289** | — |
| beam4 mnt=150 lp=1.0 | 0.4289 | ±0 |
| beam4 mnt=200 lp=1.0 | 0.4289 | ±0 |
| beam4 mnt=100 lp=1.2 | 0.4267 | -0.002 |
| beam4 mnt=100 lp=0.8 | 0.4262 | -0.003 |

**결론**: max_new_tokens 증가 효과 없음 (요약이 100 토큰 안에서 자연 종료).
length_penalty 변경 시 모두 하락. **현재 설정(beam4, mnt=100, lp=1.0)이 최적.**

### 2-B. TTA MBR vs. 샘플링 MBR 비교 — ✅ 완료

결과: beam4(0.4289) ≈ TTA 2-way(0.4288) > beam8(0.4275) >> 샘플링 MBR n=10(0.2900)

**샘플링 MBR은 KoBART에서 역효과**. beam4 단독 혹은 TTA 2-way가 최적.
TTA 오버헤드(2× 추론)는 점수 이득이 없으므로, 빠른 실험 시 beam4 단독 사용 가능.

### 2-C. TTA 3-way 확장 — ❌ 기각 (실험 완료)

짝수인덱스역전 / 홀수인덱스역전 / 랜덤셔플 3가지 변형 모두 dev에서 2-way(0.4288)보다 낮음.
발화 순서 변형 방식으로는 3-way 이득 없음. `apply_tta` 2-way 유지.

---

## 단계 3 — KoBART: 5-fold OOF 앙상블 (재학습 필요) — ✅ 완료

**5-fold 학습 결과** (`260325_run_001`):

| Fold | Best Checkpoint | Val ROUGE-1 |
|------|----------------|-------------|
| 0 | epoch11_0.7579 | 0.7579 |
| 1 | epoch11_0.7382 | 0.7382 |
| 2 | epoch06_0.7447 | 0.7447 |
| 3 | epoch12_0.7398 | 0.7398 |
| 4 | epoch09_0.7350 | 0.7350 |
| **평균** | — | **0.7431** |

- 각 fold best checkpoint → TTA 추론 → `prediction/kfold_fold{0~4}_test.csv`
- 5-fold merge → `prediction/kobart_kfold_test.csv`
- single best TTA → `prediction/kobart_single_test.csv`
- single best beam4 → `prediction/kobart_single_beam4.csv`

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

### 4-B. 프롬프트 수 최적화 — ✅ 완료

결과 (dev 100샘플, greedy max128):

**프롬프트별 단독 ROUGE (Combined 순위)**

**MBR 비교**

| 방법 | Combined | 비고 |
|------|---------|------|
| 단독 최고 (qa_style) | 0.7514 | |
| MBR 전체 8종 | 0.7294 | ❌ 단독보다 낮음 (노이즈 과다) |
| **MBR 상위 5종** (qa_style, base, base_copy, topic, narrative) | **0.7590** | ✅ 전체보다 +0.030 |

**결론**: 하위 3종(threeshot, oneshot, abstract) 제거 → **상위 5종 MBR 사용 권장**.
`LLM/prompts/mbr_prompts.py`의 `PROMPT_VARIANTS`에서 3종 주석 처리만으로 적용 가능.

### 4-D. 프롬프트 추가 및 정리 (2026-03-26)

**신규 프롬프트 3종 추가** (`LLM/prompts/mbr_prompts.py`):

| 변형 | 설명 |
|------|------|
| `gold_mimic` | Gold 정답 패턴 규칙 명시 (태그 시작, 동사 형식, 마침표 강제) |
| `observer` | 제3자 관찰자 시점으로 객관적 서술 유도 |
| `length_constrained` | 50~100자 길이 제약 명시로 간결한 요약 유도 |

**`base_copy` 비활성화**: base와 완전히 동일 → 다양성 기여 없음. 신규 3종 추가로 앙상블 다양성 확보됨.

**현재 활성 프롬프트 7종** (비활성 4종은 주석 처리):

| 상태 | 변형 |
|------|------|
| ✅ 활성 | `base`, `topic`, `narrative`, `qa_style`, `gold_mimic`, `observer`, `length_constrained` |
| 💤 비활성 | `abstract`, `oneshot`, `threeshot`, `base_copy` |

### 4-C. 후처리 점검 — ✅ 완료 (버그 없음)

- `LLM/prompts/postprocess.py:36` — `re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)` 정상 제거
- `LLM/prompts/inference.py:96` — `summary = postprocess_summary(summary)` 파이프라인 연결 확인
- 불필요한 prefix(`요약:`, `Summary:`) 제거, 화자 태그 정규화(`#Person 1#` → `#Person1#`) 정상 동작
- **버그 없음, 추가 조치 불필요**

---

## 단계 5 — Qwen: QLoRA 소규모 스윕 (학습 필요, 예상 효과: +0.5~2.0)

> 단계 4 완료 후 진행. dev 기준으로 검증된 설정만 학습.

**베이스 설정** (기존 `checkpoint-542` 기준):

| 항목 | 값 |
|------|----|
| base model | `unsloth/qwen3-14b-unsloth-bnb-4bit` |
| `lora_r` / `lora_alpha` | 32 / 32 (alpha=r, 비율 1.0) |
| `learning_rate` | 2e-4 |
| `max_seq_length` | 2048 |
| `num_epochs` | 3 (epoch 1이 best) |
| `per_device_batch` × `grad_accum` | 1 × 32 = effective bs 32 |
| Best checkpoint | epoch 1 (step 542), eval_loss=0.7034 |

### 5-A. epoch=1 스크리닝 스윕 — ✅ 완료 (A·B, C 제외)

**실행 스크립트**: `LLM/response_only_SFT/run_qlora_sweep.py`

| 실험 | `lora_r` | `lora_alpha` | `lr` | 상태 | train_loss | eval_loss |
|------|----------|--------------|------|------|-----------|-----------|
| **A** | 64 | 128 (r×2) | 2e-4 | ✅ 완료 | — | 0.7034 |
| **B** | 32 | 64 (r×2) | 1e-4 | ✅ 완료 | 0.6937 | 0.6981 |
| ~~C~~ | ~~32~~ | ~~64~~ | ~~3e-4~~ | ❌ 제외 (사용자 결정) | — | — |

- `eval_steps` = steps_per_epoch // 2 (epoch 내 2회 평가)
- 각 실험 완료 후 dev 499샘플 qa_style 단독 추론 → Combined ROUGE 자동 출력

**dev ROUGE 비교 결과** (mecab 형태소 기반, dev 499샘플):

| 실험 | R1 | R2 | RL | Combined | 비고 |
|------|-----|-----|-----|---------|------|
| A (r=64, lr=2e-4) | 0.5267 | 0.3445 | 0.4706 | 1.3417 | |
| **B (r=32, lr=1e-4)** | **0.5305** | **0.3504** | **0.4755** | **1.3563** | ✅ 최고 |

→ **B가 모든 지표에서 A를 상회** (+0.0146 Combined). **exp_B** adapter로 test 추론 진행.

- adapter 저장: `LLM/response_only_SFT/outputs/exp_B_r32_a64_lr1e4/lora_adapter/`
- test 추론 스크립트: `LLM/response_only_SFT/run_test_inference.py` (subprocess 격리 방식)
- 추론 worker: `LLM/response_only_SFT/inference_worker.py` (Unsloth 완전 차단)

> ⚠️ **Unsloth 추론 버그 기록**: `unsloth/qwen3-14b-unsloth-bnb-4bit` 모델은 Unsloth 설치 환경에서
> `FastLanguageModel.for_inference()` 호출 시 RoPE shape mismatch, `AutoModelForCausalLM` 사용 시
> `apply_qkv` 누락 오류 발생. **해결책**: `inference_worker.py`를 별도 subprocess로 실행해 Unsloth 임포트 완전 차단 후 `AutoModelForCausalLM` + `PeftModel` 사용.

### 5-B. epoch 연장 — ⏳ 대기 중

스크리닝 결과 B(r=32, lr=1e-4)가 최고. 필요 시 epoch 2~3으로 연장 예정.

---

## 단계 6 — 교차 앙상블: KoBART + Qwen — ✅ 완료

**앙상블 후보 8종 생성** (`scripts/cross_ensemble_grid.py`):

| 파일 | KoBART | Qwen | 비고 |
|------|--------|------|------|
| `ensemble_kobart0.2_qwen0.8.csv` | single TTA 0.2 | 0.8 | Qwen 우세 |
| `ensemble_kobart0.3_qwen0.7.csv` | single TTA 0.3 | 0.7 | |
| `ensemble_kobart0.4_qwen0.6.csv` | single TTA 0.4 | 0.6 | |
| `ensemble_kobart0.5_qwen0.5.csv` | single TTA 0.5 | 0.5 | |
| `ensemble_kfold0.2_qwen0.8.csv` | 5-fold merge 0.2 | 0.8 | Qwen 우세 |
| `ensemble_kfold0.3_qwen0.7.csv` | 5-fold merge 0.3 | 0.7 | |
| `ensemble_kfold0.4_qwen0.6.csv` | 5-fold merge 0.4 | 0.6 | |
| `ensemble_kfold0.5_qwen0.5.csv` | 5-fold merge 0.5 | 0.5 | |

**dev 참고 점수**:
- KoBART single dev Combined = **0.4288** (`260324_run_003/epoch06_0.7962`)
- Qwen MBR top-5 dev Combined = **0.7590** (dev 100샘플 기준)
- ⚠️ Qwen dev 예측이 없어 가중치는 dev ROUGE로 최적화하지 않음 → LB 제출 전 Qwen dev 추론 후 가중치 검증 권장

**권장 제출 순서**: `kfold0.2_qwen0.8` → `kobart0.2_qwen0.8` → `kfold0.3_qwen0.7`
(Qwen 점수가 KoBART 대비 훨씬 높으므로 Qwen 가중치 0.7~0.8이 유력)

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
