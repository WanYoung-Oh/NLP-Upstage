#!/usr/bin/env bash
# run_prompts_proper_noun.sh
# r4b_response_only_ckpt + proper_noun 포함 8종 MBR 백그라운드 실행
# 실행: bash LLM/run_prompts_proper_noun.sh [--resume]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/run_prompts_proper_noun_${TIMESTAMP}.log"
RESUME_FLAG=""

if [[ "${1:-}" == "--resume" ]]; then
    RESUME_FLAG="--resume"
fi

mkdir -p "${LOG_DIR}"

ADAPTER="${SCRIPT_DIR}/response_only_SFT/r4b_response_only_ckpt"
TEST_CSV="${SCRIPT_DIR}/response_only_SFT/data/test.csv"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 백그라운드 실행 시작 (proper_noun MBR)"
echo "  로그: ${LOG_FILE}"
echo "  어댑터: ${ADAPTER}"
echo "  테스트 CSV: ${TEST_CSV}"
echo "  프롬프트: base, topic, narrative, qa_style, gold_mimic, observer, length_constrained, proper_noun"
echo "  최종: MBR 앙상블 → prediction/proper_noun_mbr_*.csv"

nohup python "${SCRIPT_DIR}/run_prompts_proper_noun.py" \
    --adapter "${ADAPTER}" \
    --test-csv "${TEST_CSV}" \
    ${RESUME_FLAG} \
    >> "${LOG_FILE}" 2>&1 &

PID=$!
echo "  PID: ${PID}"
echo "${PID}" > "${LOG_DIR}/run_prompts_proper_noun.pid"
echo ""
echo "모니터링: tail -f ${LOG_FILE}"
echo "중단:     kill \$(cat ${LOG_DIR}/run_prompts_proper_noun.pid)"
