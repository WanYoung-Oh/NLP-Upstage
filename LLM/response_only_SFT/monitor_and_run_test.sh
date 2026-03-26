#!/bin/bash
# B 추론 완료 감지 → sweep 프로세스 종료 → test 추론 자동 실행
#
# 사용법:
#   bash monitor_and_run_test.sh
#   또는 백그라운드:
#   nohup bash monitor_and_run_test.sh > monitor.log 2>&1 &

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRED_DIR="$SCRIPT_DIR/prediction"
B_PRED_FILE="$PRED_DIR/dev_exp_B_r32_a64_lr1e4_qa_style.csv"
SWEEP_PID=3479431
CHECK_INTERVAL=60   # 1분마다 확인
LOG_FILE="$SCRIPT_DIR/monitor.log"

echo "============================================================" | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 모니터 시작" | tee -a "$LOG_FILE"
echo "  감시 파일: $B_PRED_FILE" | tee -a "$LOG_FILE"
echo "  sweep PID: $SWEEP_PID" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# B 추론 완료 대기
while true; do
    if [ -f "$B_PRED_FILE" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] B 추론 완료 감지: $B_PRED_FILE" | tee -a "$LOG_FILE"
        break
    fi

    # sweep 프로세스가 이미 종료된 경우도 확인
    if ! kill -0 "$SWEEP_PID" 2>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] sweep 프로세스(PID $SWEEP_PID) 이미 종료됨" | tee -a "$LOG_FILE"
        if [ -f "$B_PRED_FILE" ]; then
            echo "B 추론 파일 존재 → test 추론 진행" | tee -a "$LOG_FILE"
            break
        else
            echo "[오류] B 추론 파일 없고 프로세스도 없음. 수동 확인 필요." | tee -a "$LOG_FILE"
            exit 1
        fi
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 대기 중... (${CHECK_INTERVAL}초 후 재확인)" | tee -a "$LOG_FILE"
    sleep "$CHECK_INTERVAL"
done

# sweep 프로세스 종료 (C 학습 방지)
if kill -0 "$SWEEP_PID" 2>/dev/null; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] sweep 프로세스(PID $SWEEP_PID) 종료 중..." | tee -a "$LOG_FILE"
    kill "$SWEEP_PID"
    sleep 5
    if kill -0 "$SWEEP_PID" 2>/dev/null; then
        echo "SIGTERM 후 아직 살아있음 → SIGKILL" | tee -a "$LOG_FILE"
        kill -9 "$SWEEP_PID"
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] sweep 프로세스 종료 완료" | tee -a "$LOG_FILE"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] sweep 프로세스 이미 없음" | tee -a "$LOG_FILE"
fi

# GPU 메모리 완전 해제 대기
echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU 메모리 해제 대기 (30초)..." | tee -a "$LOG_FILE"
sleep 30
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | tee -a "$LOG_FILE"

# test 추론 실행
echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] test 추론 시작" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

cd "$SCRIPT_DIR"
python run_test_inference.py --batch_size 4 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 전체 파이프라인 완료!" | tee -a "$LOG_FILE"
    echo "  제출 파일 확인: $PRED_DIR/" | tee -a "$LOG_FILE"
    ls -lh "$PRED_DIR"/test_*.csv 2>/dev/null | tee -a "$LOG_FILE"
    echo "============================================================" | tee -a "$LOG_FILE"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [오류] test 추론 실패 (exit code: $EXIT_CODE)" | tee -a "$LOG_FILE"
    echo "  로그 확인: $LOG_FILE" | tee -a "$LOG_FILE"
    # batch_size=4 OOM 시 2로 재시도
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] batch_size=2로 재시도..." | tee -a "$LOG_FILE"
    python run_test_inference.py --batch_size 2 2>&1 | tee -a "$LOG_FILE"
fi
