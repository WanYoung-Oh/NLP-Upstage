#!/usr/bin/env bash
# run_prompts_qwen35.sh
# q35_final_traindev_r32_a32 LoRA + mbr_prompts 앙상블 백그라운드 실행
#
# 어댑터: response_only_SFT/outputs/q35_final_traindev_r32_a32/lora_adapter
# 프롬프트: prompts/mbr_prompts.py 의 PROMPT_VARIANTS (base, topic, qa_style, gold_mimic)
# 옵션: use_topic=true, enable_thinking=true
#
# r4b_traindev 체크포인트는 동일 폴더의 run_prompts_r4b.sh 사용
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
echo "  어댑터: response_only_SFT/outputs/q35_final_traindev_r32_a32/lora_adapter"
echo "  프롬프트: mbr_prompts (base, topic, qa_style, gold_mimic), use_topic, enable_thinking"
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
