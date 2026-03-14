#!/usr/bin/env bash
# data_aug 데이터로 4개 모델(KoT5, KoBART v2, PKO T5, Solar QLoRA)을 순차 학습합니다.
#
# SSH가 끊겨도 백그라운드에서 학습이 계속되도록 실행 방법:
#   nohup bash scripts/run_train_models_data_aug.sh &
#
# 로그는 자동으로 logs/run_train_data_aug_YYYYMMDD_HHMMSS.log 에 저장됩니다.
# 로그 확인: tail -f logs/run_train_data_aug_*.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

cd "${PROJECT_ROOT}"

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/run_train_data_aug_${TIMESTAMP}.log"

# nohup으로 실행 시 로그 경로를 터미널에 한 번 출력
echo "[run_train_models_data_aug] LOG_FILE=${LOG_FILE}"
echo "[run_train_models_data_aug] tail -f \"${LOG_FILE}\" 로 로그를 확인하세요."

exec >> "${LOG_FILE}" 2>&1

echo "=============================================="
echo "data_aug 순차 학습 시작 (${TIMESTAMP})"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "LOG_FILE=${LOG_FILE}"
echo "=============================================="

run_train() {
  local desc="$1"
  shift
  echo ""
  echo "[$(date +%H:%M:%S)] ---------- ${desc} ----------"
  python -u src/train.py general.data_path=data_aug "$@"
  echo "[$(date +%H:%M:%S)] ---------- ${desc} 완료 ----------"
}

# 1. KoT5 (T5 계열은 training=t5로 배치 축소 → 24GB GPU OOM 방지)
run_train "KoT5" model=kot5 training=t5

# 2. KoBART v2
run_train "KoBART v2" model=kobart_v2

# 3. PKO T5 (T5 계열은 training=t5로 배치 축소)
run_train "PKO T5" model=pko_t5 training=t5

# 4. Solar QLoRA
run_train "Solar QLoRA" model=solar_qlora training=qlora

echo ""
echo "=============================================="
echo "[$(date +%H:%M:%S)] 전체 순차 학습 완료."
echo "=============================================="
