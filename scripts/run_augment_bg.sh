#!/usr/bin/env bash
set -euo pipefail

# 프로젝트 루트 기준으로 경로 계산
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

# 첫 번째 인자로 method 선택 (기본값: back_translation)
METHOD="${1:-back_translation}"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/run_augment_${METHOD}_${TIMESTAMP}.log"

echo "[run_augment_bg] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[run_augment_bg] METHOD=${METHOD}"
echo "[run_augment_bg] LOG_FILE=${LOG_FILE}"

cd "${PROJECT_ROOT}"

# SSH가 끊겨도 계속 돌도록 nohup + 백그라운드 실행
nohup python -u "${PROJECT_ROOT}/src/data/run_augment.py" --method "${METHOD}" \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "[run_augment_bg] started PID=${PID}"
echo "[run_augment_bg] tail -f \"${LOG_FILE}\" 로 로그를 확인할 수 있습니다."

