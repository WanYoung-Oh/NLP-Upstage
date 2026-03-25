#!/bin/bash
# fold 학습 완료 후 자동으로 실행되는 스크립트:
#   1. 각 fold best checkpoint로 test.csv 추론
#   2. 5-fold test 예측 merge
#   3. KoBART single best TTA test 추론
#   4. KoBART × Qwen 교차 앙상블 가중치 그리드 (dev 기준)

set -e
ROOT="/data/ephemeral/home/NLP"
KFOLD_DIR="$ROOT/checkpoints/kfold"
PRED_DIR="$ROOT/prediction"
MODEL_NAME="gogamza/kobart-base-v2"
SINGLE_CKT="checkpoints/260324_run_003/epoch06_0.7962"
QWEN_PRED="$ROOT/LLM/outputs/submission_new_response_only_0324.csv"
N_FOLDS=5

cd "$ROOT"

echo "============================================================"
echo "[Step 3] 각 fold best checkpoint → test.csv 추론"
echo "============================================================"

FOLD_TEST_CSVS=()
for fold in $(seq 0 $((N_FOLDS - 1))); do
    CKPT_DIR="$KFOLD_DIR/fold_${fold}/checkpoints/260325_run_001"
    # 가장 높은 ROUGE 체크포인트 선택 (파일명: epochXX_0.XXXX)
    BEST_CKPT=$(ls "$CKPT_DIR" | sort -t_ -k2 -V | tail -1)
    FULL_CKPT="$CKPT_DIR/$BEST_CKPT"
    OUT_CSV="$PRED_DIR/kfold_fold${fold}_test.csv"

    echo "[Fold $fold] best checkpoint: $BEST_CKPT"
    python src/inference.py \
        model=kobart_v2 \
        inference=tta \
        inference.ckt_path="$FULL_CKPT" \
        inference.result_path="$PRED_DIR" \
        inference.output_filename="kfold_fold${fold}_test.csv" \
        general.data_path="data"

    FOLD_TEST_CSVS+=("$OUT_CSV")
done

echo ""
echo "============================================================"
echo "[Step 3] 5-fold test 예측 merge → kobart_kfold_test.csv"
echo "============================================================"
python src/ensemble_cli.py merge \
    --inputs "${FOLD_TEST_CSVS[@]}" \
    --output "$PRED_DIR/kobart_kfold_test.csv"
echo "저장: $PRED_DIR/kobart_kfold_test.csv"

echo ""
echo "============================================================"
echo "[Step 3] single best KoBART TTA test 추론"
echo "============================================================"
python src/inference.py \
    model=kobart_v2 \
    inference=tta \
    inference.ckt_path="$SINGLE_CKT" \
    inference.result_path="$PRED_DIR" \
    inference.output_filename="kobart_single_test.csv" \
    general.data_path="data"
echo "저장: $PRED_DIR/kobart_single_test.csv"

echo ""
echo "============================================================"
echo "[Step 5] single best KoBART beam4 test 추론"
echo "============================================================"
python src/inference.py \
    model=kobart_v2 \
    inference=beam4 \
    inference.ckt_path="$SINGLE_CKT" \
    inference.result_path="$PRED_DIR" \
    inference.output_filename="kobart_single_beam4.csv" \
    general.data_path="data"
echo "저장: $PRED_DIR/kobart_single_beam4.csv"

echo ""
echo "============================================================"
echo "[Step 6] KoBART × Qwen 교차 앙상블 가중치 그리드"
echo "============================================================"
python scripts/cross_ensemble_grid.py \
    --kobart_single "$PRED_DIR/kobart_single_test.csv" \
    --kobart_kfold "$PRED_DIR/kobart_kfold_test.csv" \
    --qwen "$QWEN_PRED" \
    --dev_csv "data/dev.csv" \
    --ckt_path "$SINGLE_CKT" \
    --model_name "$MODEL_NAME" \
    --output_dir "$PRED_DIR"

echo ""
echo "============================================================"
echo "전체 완료"
echo "============================================================"
