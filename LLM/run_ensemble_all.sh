#!/usr/bin/env bash
# run_ensemble_all.sh
# Mode 1 → Mode 2 recommended → Mode 3 순차 실행
# 각 모드 완료 후 submission 파일 생성
# 실행: bash run_ensemble_all.sh [--resume]
# 로그: logs/ensemble_all_<timestamp>.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_FILE="${SCRIPT_DIR}/response_only_SFT/data/test.csv"
PRED_DIR="/data/ephemeral/home/prediction"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/ensemble_all_${TIMESTAMP}.log"
RESUME_FLAG=""

if [[ "${1:-}" == "--resume" ]]; then
    RESUME_FLAG="--resume"
fi

mkdir -p "${LOG_DIR}" "${PRED_DIR}"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "${msg}" | tee -a "${LOG_FILE}"
}

run_mode() {
    local mode_name="$1"
    local config="$2"
    local output_csv="$3"

    log "========================================================"
    log "시작: ${mode_name}"
    log "  config : ${config}"
    log "  output : ${output_csv}"
    log "========================================================"

    local start_ts
    start_ts=$(date +%s)

    python "${SCRIPT_DIR}/run_ensemble.py" \
        --config "${config}" \
        --test_file "${TEST_FILE}" \
        --output_file "${output_csv}" \
        ${RESUME_FLAG} \
        2>&1 | tee -a "${LOG_FILE}"

    local exit_code=${PIPESTATUS[0]}
    local elapsed=$(( $(date +%s) - start_ts ))
    local hms
    hms=$(printf '%02dh %02dm %02ds' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)))

    if [[ ${exit_code} -ne 0 ]]; then
        log "[오류] ${mode_name} 실패 (exit=${exit_code}, elapsed=${hms})"
        exit ${exit_code}
    fi

    log "완료: ${mode_name} | 소요: ${hms} | 저장: ${output_csv}"
}

log "앙상블 전체 실행 시작 (TIMESTAMP=${TIMESTAMP})"
log "  TEST_FILE : ${TEST_FILE}"
log "  PRED_DIR  : ${PRED_DIR}"
log "  LOG_FILE  : ${LOG_FILE}"

# ── Mode 1 ──────────────────────────────────────────────────
run_mode \
    "Mode 1 (quick: 3 ckpt × 1 prompt)" \
    "${SCRIPT_DIR}/conf/ensemble_mode1_quick.yaml" \
    "${PRED_DIR}/ensemble_mode1_${TIMESTAMP}.csv"

# ── Mode 2 recommended ──────────────────────────────────────
run_mode \
    "Mode 2 recommended (3 ckpt × 7 prompts)" \
    "${SCRIPT_DIR}/conf/ensemble_mode2_recommended.yaml" \
    "${PRED_DIR}/ensemble_mode2_recommended_${TIMESTAMP}.csv"

# ── Mode 3 sampling ─────────────────────────────────────────
run_mode \
    "Mode 3 sampling (3 ckpt × 3 prompts × greedy+sampling)" \
    "${SCRIPT_DIR}/conf/ensemble_mode3_sampling.yaml" \
    "${PRED_DIR}/ensemble_mode3_sampling_${TIMESTAMP}.csv"

log "========================================================"
log "전체 완료. 생성된 서브미션 파일:"
log "  ${PRED_DIR}/ensemble_mode1_${TIMESTAMP}.csv"
log "  ${PRED_DIR}/ensemble_mode2_recommended_${TIMESTAMP}.csv"
log "  ${PRED_DIR}/ensemble_mode3_sampling_${TIMESTAMP}.csv"
log "========================================================"
