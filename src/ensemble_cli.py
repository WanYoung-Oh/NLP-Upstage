#!/usr/bin/env python3
"""
앙상블 CLI (src/ensemble.py 연동).

예시:
  python src/ensemble_cli.py merge \\
    --inputs prediction/a.csv prediction/b.csv \\
    --output prediction/ensemble_out.csv
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from src.ensemble import (
    WeightedEnsemble,
    list_checkpoints,
    select_checkpoints_for_ensemble,
)


def _align_prediction_frames(dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """fname 교집합으로 정렬해 행 순서를 맞춥니다."""
    if len(dfs) < 2:
        raise ValueError("merge에는 예측 CSV가 최소 2개 필요합니다.")
    required = {"fname", "summary"}
    for i, df in enumerate(dfs):
        miss = required - set(df.columns)
        if miss:
            raise ValueError(f"{i}번째 CSV에 컬럼 없음: {miss} (필요: fname, summary)")

    common = set(dfs[0]["fname"])
    for df in dfs[1:]:
        common &= set(df["fname"])
    if not common:
        raise ValueError("모든 CSV에 공통으로 있는 fname이 없습니다.")

    order = [f for f in dfs[0]["fname"] if f in common]
    aligned: list[pd.DataFrame] = []
    for df in dfs:
        by_f = df.set_index("fname")["summary"]
        aligned.append(
            pd.DataFrame({"fname": order, "summary": [by_f[f] for f in order]})
        )
    return aligned


def cmd_merge(args: argparse.Namespace) -> None:
    dfs = [pd.read_csv(p) for p in args.inputs]
    aligned = _align_prediction_frames(dfs)
    if len(aligned[0]) < len(dfs[0]):
        n_drop = len(dfs[0]) - len(aligned[0])
        print(f"[merge] fname 교집합만 사용합니다. 첫 CSV 기준 {n_drop}행 제외.", file=sys.stderr)

    weights = args.weights
    oof_scores = args.oof_scores
    if weights is not None and oof_scores is not None:
        raise SystemExit("--weights 와 --oof-scores 는 동시에 쓸 수 없습니다.")
    if weights is not None and len(weights) != len(aligned):
        raise SystemExit(f"--weights 개수({len(weights)})가 입력 CSV 개수({len(aligned)})와 같아야 합니다.")
    if oof_scores is not None and len(oof_scores) != len(aligned):
        raise SystemExit(f"--oof-scores 개수({len(oof_scores)})가 입력 CSV 개수({len(aligned)})와 같아야 합니다.")

    ens = WeightedEnsemble()
    out = ens.predict(aligned, weights=weights, oof_scores=oof_scores)
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved → {args.output} ({len(out)} rows)")


def cmd_list_checkpoints(args: argparse.Namespace) -> None:
    rows = list_checkpoints(args.root, run_id=args.run_id)
    if not rows:
        print("(체크포인트 없음)", file=sys.stderr)
        return
    for r in rows:
        print(
            f"{r['run_id']}\tepoch{r['epoch']:02d}\tscore={r['score']:.4f}\t{r['path']}"
        )


def cmd_select_checkpoints(args: argparse.Namespace) -> None:
    run_ids = args.runs if args.runs else None
    paths = select_checkpoints_for_ensemble(
        args.root,
        run_ids=run_ids,
        top_k_per_run=args.top_k_per_run,
        min_score=args.min_score,
    )
    for p in paths:
        print(p)
    if not paths:
        print("(선택된 경로 없음)", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="예측 CSV 앙상블 및 체크포인트 탐색 (ensemble.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_merge = sub.add_parser("merge", help="여러 prediction CSV → WeightedEnsemble")
    p_merge.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        metavar="CSV",
        help="fname, summary 컬럼이 있는 예측 파일 (2개 이상)",
    )
    p_merge.add_argument("--output", "-o", required=True, help="저장할 CSV 경로")
    p_merge.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="모델별 가중치 (개수 = 입력 CSV 개수)",
    )
    p_merge.add_argument(
        "--oof-scores",
        nargs="+",
        type=float,
        default=None,
        metavar="SCORE",
        help="OOF 등 점수 — weights 미지정 시 정규화해 사용",
    )
    p_merge.set_defaults(func=cmd_merge)

    p_list = sub.add_parser("list-checkpoints", help="checkpoints/ 아래 epoch 폴더 나열")
    p_list.add_argument("--root", default="checkpoints", help="checkpoints 루트")
    p_list.add_argument("--run-id", default=None, help="특정 run_id만 (예: 260314_run_005)")
    p_list.set_defaults(func=cmd_list_checkpoints)

    p_sel = sub.add_parser(
        "select-checkpoints",
        help="앙상블용 체크포인트 경로만 출력 (한 줄에 하나)",
    )
    p_sel.add_argument("--root", default="checkpoints")
    p_sel.add_argument(
        "--runs",
        nargs="+",
        default=None,
        metavar="RUN_ID",
        help="포함할 run_id 목록. 생략 시 전체 run",
    )
    p_sel.add_argument("--top-k-per-run", type=int, default=1)
    p_sel.add_argument("--min-score", type=float, default=0.0)
    p_sel.set_defaults(func=cmd_select_checkpoints)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()