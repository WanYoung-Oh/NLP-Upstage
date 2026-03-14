"""
Phase 5: 앙상블 및 GroupKFold OOF 모듈.

- GroupKFoldTrainer: topic 그룹 기반 5-fold CV OOF 학습
- WeightedEnsemble: 다수 모델 예측의 가중치 앙상블
- MBRDecoder: N개 후보 중 평균 ROUGE 최고 선택
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from tqdm import tqdm


# ---------------------------------------------------------------------------
# MBR Decoder (재사용 가능한 독립 모듈)
# ---------------------------------------------------------------------------

class MBRDecoder:
    """N개 요약 후보 중 다른 후보들과 평균 ROUGE가 가장 높은 것을 선택."""

    def decode(self, candidates: list[str]) -> str:
        if not candidates:
            return ""

        from rouge import Rouge

        rouge = Rouge()
        best, best_score = candidates[0], -1.0
        for i, cand in enumerate(candidates):
            others = [c for j, c in enumerate(candidates) if j != i and c.strip()]
            if not others:
                continue
            cand_safe = cand.strip() if cand.strip() else "."
            try:
                scores = rouge.get_scores([cand_safe] * len(others), others, avg=True)
                avg = scores["rouge-l"]["f"]
            except Exception:
                avg = 0.0
            if avg > best_score:
                best_score = avg
                best = cand
        return best


# ---------------------------------------------------------------------------
# Weighted Ensemble
# ---------------------------------------------------------------------------

class WeightedEnsemble:
    """
    다수 모델 예측 CSV를 ROUGE 기반 가중치로 앙상블합니다.

    가중치 미지정 시 OOF ROUGE 점수를 기반으로 자동 계산합니다.
    """

    def predict(
        self,
        predictions_list: list[pd.DataFrame],
        weights: list[float] | None = None,
        oof_scores: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Args:
            predictions_list: 각 모델의 예측 DataFrame (fname, summary 컬럼)
            weights: 명시적 가중치 (합산=1이 아니어도 자동 정규화)
            oof_scores: OOF ROUGE 점수 (weights 미지정 시 사용)

        Returns:
            앙상블 결과 DataFrame (fname, summary)
        """
        if len(predictions_list) < 2:
            raise ValueError(f"최소 2개 모델 예측 필요, 현재 {len(predictions_list)}개")

        if weights is None:
            if oof_scores is not None:
                total = sum(oof_scores)
                weights = [s / total for s in oof_scores]
            else:
                weights = [1.0 / len(predictions_list)] * len(predictions_list)

        # 정규화
        total = sum(weights)
        weights = [w / total for w in weights]

        # MBR 기반 최종 선택
        decoder = MBRDecoder()
        base_df = predictions_list[0][["fname"]].copy()
        summaries_matrix = [df["summary"].tolist() for df in predictions_list]

        final_summaries = []
        for i in tqdm(range(len(base_df)), desc="Ensemble"):
            candidates = [summaries_matrix[m][i] for m in range(len(predictions_list))]
            # 가중치에 따라 후보 복제 (정수 반올림)
            weighted_candidates = []
            for cand, w in zip(candidates, weights):
                copies = max(1, round(w * 10))
                weighted_candidates.extend([cand] * copies)
            final_summaries.append(decoder.decode(weighted_candidates))

        base_df["summary"] = final_summaries
        return base_df


# ---------------------------------------------------------------------------
# GroupKFold OOF Trainer
# ---------------------------------------------------------------------------

def _find_best_checkpoint(checkpoint_root: str) -> str | None:
    """
    checkpoint_root/{run_id}/epoch##_score 구조에서 점수가 가장 높은 폴더를 반환합니다.

    BestCheckpointCallback 저장 규칙:
        {checkpoints_root}/{yymmdd_run_NNN}/epoch{##}_{score:.4f}

    train.py는 general.checkpoints_root 아래에 run_id 디렉토리를 생성하므로,
    이 함수는 한 단계 더 내려가 run_id 하위에서 epoch 폴더를 탐색합니다.
    """
    import re

    run_pattern = re.compile(r"^\d{6}_run_\d+$")
    epoch_pattern = re.compile(r"^epoch\d+_([\d.]+)$")
    best_path, best_score = None, -1.0
    if not os.path.isdir(checkpoint_root):
        return None
    for run_dir_name in os.listdir(checkpoint_root):
        run_dir = os.path.join(checkpoint_root, run_dir_name)
        if not (run_pattern.match(run_dir_name) and os.path.isdir(run_dir)):
            continue
        for name in os.listdir(run_dir):
            m = epoch_pattern.match(name)
            if m:
                score = float(m.group(1))
                if score > best_score:
                    best_score = score
                    best_path = os.path.join(run_dir, name)
    return best_path


class GroupKFoldTrainer:
    """
    topic 그룹 기반 GroupKFold CV OOF 학습.

    각 fold에서 best checkpoint를 저장하고 OOF 예측을 생성합니다.
    train_oof()는 다음 단계를 자동으로 수행합니다:
      1. fold별 train/val CSV 생성
      2. src/train.py subprocess 호출로 학습
      3. best checkpoint를 탐색해 src/inference.py로 val fold 추론
      4. OOF 예측 DataFrame 반환
    """

    def __init__(self, n_splits: int = 5) -> None:
        self.n_splits = n_splits

    def train_oof(
        self,
        train_csv: str,
        cfg_overrides: list[str] | None = None,
        output_dir: str = "checkpoints/kfold",
    ) -> pd.DataFrame:
        """
        Args:
            train_csv: train.csv 경로
            cfg_overrides: Hydra 스타일 override 리스트 (예: ["model=kobart", "training=baseline"])
            output_dir: fold별 checkpoint 및 예측 저장 루트 디렉토리

        Returns:
            OOF 예측 DataFrame (fname, summary, fold)
        """
        df = pd.read_csv(train_csv)
        gkf = GroupKFold(n_splits=self.n_splits)
        groups = df["topic"].fillna("unknown").tolist()

        oof_frames: list[pd.DataFrame] = []

        for fold, (train_idx, val_idx) in enumerate(
            gkf.split(df.index, groups=groups)
        ):
            print(f"\n{'='*60}")
            print(f"Fold {fold + 1}/{self.n_splits}")
            print(f"{'='*60}")

            fold_train = df.iloc[train_idx]
            fold_val = df.iloc[val_idx]

            fold_data_dir = os.path.join(output_dir, f"fold_{fold}", "data")
            fold_ckpt_dir = os.path.join(output_dir, f"fold_{fold}", "checkpoints")
            fold_pred_dir = os.path.join(output_dir, f"fold_{fold}", "predictions")
            os.makedirs(fold_data_dir, exist_ok=True)

            fold_train.to_csv(os.path.join(fold_data_dir, "train.csv"), index=False)
            fold_val.to_csv(os.path.join(fold_data_dir, "dev.csv"), index=False)
            # inference는 test.csv를 읽으므로 val을 test.csv로도 저장
            fold_val[["fname", "dialogue"]].to_csv(
                os.path.join(fold_data_dir, "test.csv"), index=False
            )

            print(f"train={len(fold_train)}, val={len(fold_val)}")

            # ── 1. 학습 ──────────────────────────────────────────────
            # general.checkpoints_root를 fold별 디렉토리로 override해
            # train.py가 저장하는 경로와 _find_best_checkpoint가 탐색하는 경로를 일치시킵니다.
            train_cmd = [
                sys.executable, "src/train.py",
                f"general.data_path={fold_data_dir}",
                f"general.checkpoints_root={fold_ckpt_dir}",
                *(cfg_overrides or []),
            ]
            print(f"[Fold {fold}] 학습 시작: {' '.join(train_cmd)}")
            subprocess.run(train_cmd, check=True)

            # ── 2. best checkpoint 탐색 ──────────────────────────────
            # train.py는 fold_ckpt_dir/{run_id}/epoch##_score/ 구조로 저장합니다.
            best_ckpt = _find_best_checkpoint(fold_ckpt_dir)
            if best_ckpt is None:
                print(f"[Fold {fold}] WARNING: best checkpoint를 찾지 못했습니다. 건너뜁니다.")
                continue

            # ── 3. OOF 추론 ─────────────────────────────────────────
            infer_cmd = [
                sys.executable, "src/inference.py",
                f"general.data_path={fold_data_dir}",
                f"inference.ckt_path={best_ckpt}",
                f"inference.result_path={fold_pred_dir}",
                f"inference.output_filename=oof_fold{fold}.csv",
            ]
            print(f"[Fold {fold}] 추론 시작: {' '.join(infer_cmd)}")
            subprocess.run(infer_cmd, check=True)

            # ── 4. OOF 결과 수집 ─────────────────────────────────────
            pred_path = os.path.join(fold_pred_dir, f"oof_fold{fold}.csv")
            if not os.path.exists(pred_path):
                print(f"[Fold {fold}] WARNING: 예측 파일 없음: {pred_path}")
                continue

            fold_pred = pd.read_csv(pred_path)
            fold_pred["fold"] = fold
            oof_frames.append(fold_pred)

        if not oof_frames:
            return pd.DataFrame(columns=["fname", "summary", "fold"])

        return pd.concat(oof_frames, ignore_index=True)
