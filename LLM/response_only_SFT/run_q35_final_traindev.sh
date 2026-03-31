#!/usr/bin/env bash
# run_q35_final_traindev.sh
# Qwen3.5-9B 최종 실험 (train+dev 병합, Topic, enable_thinking=True) 백그라운드 실행
#
# 실행: bash LLM/response_only_SFT/run_q35_final_traindev.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/run_q35_final_traindev_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 백그라운드 실행 시작 (q35_final_traindev)"
echo "  로그: ${LOG_FILE}"
echo "  모델: Qwen/Qwen3.5-9B"
echo "  설정: LoRA R=32/alpha=32, epochs=3, train+dev 병합, Topic, enable_thinking=True"
echo "  출력: prediction/q35_final_traindev_*.csv"

nohup python "${SCRIPT_DIR}/run_q35_final_traindev.py" \
    >> "${LOG_FILE}" 2>&1 &

PID=$!
echo "  PID: ${PID}"
echo "${PID}" > "${LOG_DIR}/run_q35_final_traindev.pid"
echo ""
echo "모니터링: tail -f ${LOG_FILE}"
echo "중단:     kill \$(cat ${LOG_DIR}/run_q35_final_traindev.pid)"
