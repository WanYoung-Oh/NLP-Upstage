#!/usr/bin/env bash
# run_prompts_qwen35.sh
# qwen35_9b_lora_sft 프롬프트 앙상블 백그라운드 실행
#
# 실행: bash run_prompts_qwen35.sh [--resume]
# 로그: logs/run_prompts_qwen35_<timestamp>.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/run_prompts_qwen35_${TIMESTAMP}.log"
RESUME_FLAG=""

if [[ "${1:-}" == "--resume" ]]; then
    RESUME_FLAG="--resume"
fi

mkdir -p "${LOG_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 백그라운드 실행 시작"
echo "  로그: ${LOG_FILE}"
echo "  프롬프트: base, topic, qa_style, gold_mimic"
echo "  최종: MBR 앙상블"

nohup python "${SCRIPT_DIR}/run_prompts_qwen35.py" \
    ${RESUME_FLAG} \
    >> "${LOG_FILE}" 2>&1 &

PID=$!
echo "  PID: ${PID}"
echo "${PID}" > "${LOG_DIR}/run_prompts_qwen35.pid"
echo ""
echo "모니터링: tail -f ${LOG_FILE}"
echo "중단:     kill \$(cat ${LOG_DIR}/run_prompts_qwen35.pid)"
