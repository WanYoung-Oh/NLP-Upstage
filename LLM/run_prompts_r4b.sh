#!/usr/bin/env bash
# run_prompts_r4b.sh
# r4b_traindev_ckpt LoRA + response_only_SFT/data/test.csv 프롬프트 앙상블 백그라운드 실행
#
# 실행: bash run_prompts_r4b.sh [--resume]
# 로그: LLM/logs/run_prompts_r4b_<timestamp>.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/run_prompts_r4b_${TIMESTAMP}.log"
RESUME_FLAG=""

if [[ "${1:-}" == "--resume" ]]; then
    RESUME_FLAG="--resume"
fi

mkdir -p "${LOG_DIR}"

ADAPTER="${SCRIPT_DIR}/response_only_SFT/outputs/r4b_traindev_ckpt/lora_adapter"
TEST_CSV="${SCRIPT_DIR}/response_only_SFT/data/test.csv"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 백그라운드 실행 시작 (r4b_traindev)"
echo "  로그: ${LOG_FILE}"
echo "  어댑터: ${ADAPTER}"
echo "  테스트 CSV: ${TEST_CSV}"
echo "  프롬프트: base, topic, qa_style, gold_mimic"
echo "  최종: MBR 앙상블 → prediction/r4b_mbr_*.csv"

nohup python "${SCRIPT_DIR}/run_prompts_r4b.py" \
    --adapter "${ADAPTER}" \
    --test-csv "${TEST_CSV}" \
    ${RESUME_FLAG} \
    >> "${LOG_FILE}" 2>&1 &

PID=$!
echo "  PID: ${PID}"
echo "${PID}" > "${LOG_DIR}/run_prompts_r4b.pid"
echo ""
echo "모니터링: tail -f ${LOG_FILE}"
echo "중단:     kill \$(cat ${LOG_DIR}/run_prompts_r4b.pid)"
