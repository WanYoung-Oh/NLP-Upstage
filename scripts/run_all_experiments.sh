#!/bin/bash
# =============================================================================
# 전체 실험 자동화 스크립트 — PRD Phase 1~5 실험 커버
#
# 실행 방법:
#   bash scripts/run_all_experiments.sh
#
# 개별 Phase만 실행하려면 해당 함수를 직접 호출:
#   source scripts/run_all_experiments.sh && run_phase2
# =============================================================================

set -e
cd "$(dirname "$(dirname "$(realpath "$0")")")"

# .env 파일 자동 로드
if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*" | tee -a "$LOG_DIR/run_all.log"; }

# =============================================================================
# Phase 1 — 베이스라인 재현
# =============================================================================
run_phase1() {
    log "=== Phase 1: 베이스라인 재현 ==="

    # 1-1. 학습
    log "[Phase1] KoBART 베이스라인 학습"
    python src/train.py \
        model=kobart \
        training=baseline \
        2>&1 | tee "$LOG_DIR/phase1_train.log"

    # 1-2. 추론 (best checkpoint 자동 탐색)
    BEST_CKT=$(python -c "
import sys; sys.path.insert(0, '.')
from scripts.evaluate_on_dev import find_best_checkpoint
print(find_best_checkpoint() or '')
")
    if [ -z "$BEST_CKT" ]; then
        log "[Phase1] 체크포인트를 찾지 못했습니다."
        return 1
    fi
    log "[Phase1] Best checkpoint: $BEST_CKT"

    python src/inference.py \
        inference.ckt_path="$BEST_CKT" \
        inference=beam4 \
        2>&1 | tee "$LOG_DIR/phase1_inference.log"

    log "[Phase1] 완료 — prediction/output.csv 생성 확인"
    ls -lh prediction/output.csv
}

# =============================================================================
# Phase 2 — 모델 업그레이드
# =============================================================================
run_phase2() {
    log "=== Phase 2: 모델 업그레이드 ==="

    # 2-1. KoT5
    # log "[Phase2] KoT5-summarization 학습"
    # python src/train.py model=kot5 training=baseline \
    #     2>&1 | tee "$LOG_DIR/phase2_kot5.log"

    # 2-2. kobart-v2
    log "[Phase2] kobart-base-v2 학습"
    python src/train.py model=kobart_v2 training=baseline \
        2>&1 | tee "$LOG_DIR/phase2_kobart_v2.log"

    # 2-3. pko-T5-large
    # log "[Phase2] pko-T5-large 학습"
    # python src/train.py model=pko_t5 training=full \
    #     2>&1 | tee "$LOG_DIR/phase2_pko_t5.log"

    # 2-4. Hydra sweep (3모델 동시)
    # log "[Phase2] Hydra sweep: kobart, kot5, pko_t5"
    # python src/train.py -m model=kobart,kot5,pko_t5 training=baseline \
    #     2>&1 | tee "$LOG_DIR/phase2_sweep.log"

    # (선택) SOLAR QLoRA — 고사양 GPU 필요
    # python src/train.py model=solar_qlora training=qlora \
    #     2>&1 | tee "$LOG_DIR/phase2_solar.log"

    log "[Phase2] 완료"
}

# =============================================================================
# Phase 3 — 데이터/학습 전략 고도화
# =============================================================================
run_phase3() {
    log "=== Phase 3: 데이터/학습 전략 고도화 ==="

    # 3-1. EDA 증강 데이터 생성
    log "[Phase3] EDA 증강 데이터 생성"
    python src/data/run_augment.py \
        --method eda \
        --max_samples 3000 \
        --rouge_threshold 0.3 \
        --data_dir data \
        --output_dir data_aug \
        2>&1 | tee "$LOG_DIR/phase3_augment_eda.log"

    # (선택) Back-translation — googletrans API 필요
    # python src/data/run_augment.py --method back_translation \
    #     2>&1 | tee "$LOG_DIR/phase3_augment_bt.log"

    # 3-2. 클리닝 + 필터 활성화로 재학습
    log "[Phase3] 클리닝+필터 활성화 학습"
    python src/train.py \
        model=kobart_v2 \
        training=full \
        data.use_cleaning=true \
        data.use_length_filter=true \
        2>&1 | tee "$LOG_DIR/phase3_cleaning.log"

    # 3-3. 증강 데이터로 학습 (EDA)
    log "[Phase3] EDA 증강 데이터로 학습"
    python src/train.py \
        model=kobart_v2 \
        training=full \
        general.data_path=data_aug \
        2>&1 | tee "$LOG_DIR/phase3_aug_train.log"

    # 3-4. Train+Dev 합산 학습 (최종 제출 전)
    log "[Phase3] Train+Dev 합산 학습 (use_all_data=true)"
    BEST_CKT=$(python -c "
import sys; sys.path.insert(0, '.')
from scripts.evaluate_on_dev import find_best_checkpoint
print(find_best_checkpoint() or '')
")
    BEST_EPOCHS=$(basename "$BEST_CKT" | grep -oP '(?<=epoch)\d+')
    log "[Phase3] best epoch: $BEST_EPOCHS"
    python src/train.py \
        model=kobart_v2 \
        training=full \
        training.use_all_data=true \
        training.num_train_epochs="$BEST_EPOCHS" \
        2>&1 | tee "$LOG_DIR/phase3_all_data.log"

    log "[Phase3] 완료"
}

# =============================================================================
# Phase 4 — Solar API
# =============================================================================
run_phase4() {
    log "=== Phase 4: Solar API 추론 ==="

    # 4-1. Zero-shot 추론
    log "[Phase4] Zero-shot 추론 (dev 100개)"
    python -c "
import os, sys, pandas as pd
sys.path.insert(0, '.')
os.environ.setdefault('UPSTAGE_API_KEY', os.environ.get('UPSTAGE_API_KEY', ''))
from src.inference import SolarAPIInferencer
from omegaconf import OmegaConf
cfg = OmegaConf.load('conf/inference/zero_shot_solar.yaml')
# dev 100개 샘플 추론
import shutil
dev = pd.read_csv('data/dev.csv').head(100)
dev[['fname','dialogue']].to_csv('/tmp/test_zero_shot.csv', index=False)
# inference.py는 test.csv 읽으므로 임시 복사
" 2>&1 | tee "$LOG_DIR/phase4_zero_shot.log"

    python src/inference.py inference=zero_shot_solar \
        2>&1 | tee -a "$LOG_DIR/phase4_zero_shot.log"

    # 4-2. Few-shot 추론 (전체 test)
    log "[Phase4] Few-shot 추론 (전체 test.csv)"
    python src/inference.py inference=solar_api \
        2>&1 | tee "$LOG_DIR/phase4_few_shot.log"

    log "[Phase4] 완료"
}

# =============================================================================
# Phase 5 — 앙상블 & 후처리 & 비교
# =============================================================================
run_phase5() {
    log "=== Phase 5: 앙상블 & 성능 비교 ==="

    BEST_CKT=$(python -c "
import sys; sys.path.insert(0, '.')
from scripts.evaluate_on_dev import find_best_checkpoint
print(find_best_checkpoint() or '')
")

    if [ -z "$BEST_CKT" ]; then
        log "[Phase5] 체크포인트를 찾지 못했습니다."
        return 1
    fi

    # 5-1. 후처리 전/후 + beam4 vs beam8 vs MBR + TTA 비교
    log "[Phase5] beam4 vs beam8 vs MBR vs TTA 전체 비교"
    python scripts/evaluate_on_dev.py \
        --ckt_path "$BEST_CKT" \
        --run_all \
        --n_tta_ways 2 \
        --output_csv prediction/dev_eval_results.csv \
        2>&1 | tee "$LOG_DIR/phase5_comparison.log"

    # 5-2. beam8 추론 → test 예측
    log "[Phase5] beam8 test 추론"
    python src/inference.py \
        inference=beam8 \
        inference.ckt_path="$BEST_CKT" \
        inference.output_filename=output_beam8.csv \
        2>&1 | tee "$LOG_DIR/phase5_beam8_infer.log"

    # 5-3. TTA 추론 → test 예측
    log "[Phase5] TTA test 추론"
    python src/inference.py \
        inference=tta \
        inference.ckt_path="$BEST_CKT" \
        inference.output_filename=output_tta.csv \
        2>&1 | tee "$LOG_DIR/phase5_tta_infer.log"

    # 5-4. 앙상블 (beam4 + beam8 + TTA)
    log "[Phase5] 앙상블 — beam4 + beam8 + TTA"
    python -c "
import sys, pandas as pd
sys.path.insert(0, '.')
from src.ensemble import WeightedEnsemble, MBRDecoder

pred_beam4 = pd.read_csv('prediction/output.csv')
pred_beam8 = pd.read_csv('prediction/output_beam8.csv')
pred_tta   = pd.read_csv('prediction/output_tta.csv')

ensemble = WeightedEnsemble()
result = ensemble.predict(
    [pred_beam4, pred_beam8, pred_tta],
    weights=[0.4, 0.35, 0.25],
)
result.to_csv('prediction/output_ensemble.csv', index=False)
print(f'앙상블 완료: {len(result)}건 → prediction/output_ensemble.csv')
" 2>&1 | tee "$LOG_DIR/phase5_ensemble.log"

    log "[Phase5] 완료"
}

# =============================================================================
# 메인 진입점
# =============================================================================
main() {
    case "${1:-all}" in
        phase1) run_phase1 ;;
        phase2) run_phase2 ;;
        phase3) run_phase3 ;;
        phase4) run_phase4 ;;
        phase5) run_phase5 ;;
        all)
            run_phase1
            run_phase2
            run_phase3
            # run_phase4  # API key 필요
            run_phase5
            ;;
        *)
            echo "사용법: bash scripts/run_all_experiments.sh [phase1|phase2|phase3|phase4|phase5|all]"
            exit 1
            ;;
    esac
    log "=== 전체 완료 ==="
}

main "$@"
