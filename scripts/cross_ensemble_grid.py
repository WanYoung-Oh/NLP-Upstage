"""
KoBART × Qwen 교차 앙상블 가중치 그리드 (단계 6)

dev.csv에서 KoBART dev 예측을 생성한 뒤
다양한 가중치로 앙상블하여 최적 가중치를 탐색합니다.

실행 (auto_ensemble_after_kfold.sh에서 자동 호출):
    python scripts/cross_ensemble_grid.py \
        --kobart_single prediction/kobart_single_test.csv \
        --kobart_kfold  prediction/kobart_kfold_test.csv \
        --qwen          LLM/outputs/submission_new_response_only_0324.csv \
        --dev_csv       data/dev.csv \
        --ckt_path      checkpoints/260324_run_003/epoch06_0.7962 \
        --model_name    gogamza/kobart-base-v2 \
        --output_dir    prediction
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.metrics import _rouge_baseline
from src.utils.postprocess import batch_postprocess
from src.ensemble import WeightedEnsemble


def _rouge(preds, refs):
    s = _rouge_baseline(
        [p if p.strip() else "." for p in preds],
        [r if r.strip() else "." for r in refs],
    )
    return s["rouge-1"] + s["rouge-2"] + s["rouge-l"]


def generate_kobart_dev_preds(ckt_path: str, model_name: str,
                               dev_df: pd.DataFrame, batch_size: int = 8) -> list[str]:
    """KoBART로 dev 대화에 대한 요약 생성 (TTA 2-way)."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from torch.utils.data import DataLoader
    from src.data.preprocess import DatasetForInference, apply_tta
    from src.ensemble import MBRDecoder
    from src.utils.device import get_device
    from tqdm import tqdm

    special_tokens = [
        "#Person1#", "#Person2#", "#Person3#", "#Person4#",
        "#Person5#", "#Person6#", "#Person7#",
        "#PhoneNumber#", "#Address#", "#PassportNumber#",
        "#DateOfBirth#", "#SSN#", "#CardNumber#", "#CarNumber#", "#Email#",
    ]
    resolved = ckt_path if os.path.isabs(ckt_path) else str(_ROOT / ckt_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model = AutoModelForSeq2SeqLM.from_pretrained(resolved)
    model.resize_token_embeddings(len(tokenizer))
    device = get_device()
    model.to(device).eval()

    dialogues = dev_df["dialogue"].tolist()
    tta_variants = apply_tta(dialogues, n_ways=2)
    mbr = MBRDecoder()
    remove_tokens = ["<s>", "</s>", "<pad>", "<unk>"]
    preds = []

    print(f"  KoBART dev TTA 추론 중 ({len(dialogues)}건)...")
    for variants in tqdm(tta_variants, desc="KoBART dev TTA"):
        cands = []
        for v in variants:
            tok = tokenizer(
                [v], return_tensors="pt", padding=True,
                truncation=True, max_length=512,
                add_special_tokens=True, return_token_type_ids=False,
            )
            with torch.no_grad():
                gen = model.generate(
                    input_ids=tok["input_ids"].to(device),
                    attention_mask=tok["attention_mask"].to(device),
                    num_beams=4, max_new_tokens=100,
                    no_repeat_ngram_size=3, early_stopping=True,
                )
            cands.append(tokenizer.decode(gen[0], skip_special_tokens=True))
        preds.append(mbr.decode(cands))

    preds = batch_postprocess(preds, remove_tokens)
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kobart_single", required=True)
    parser.add_argument("--kobart_kfold", default=None)
    parser.add_argument("--qwen", required=True)
    parser.add_argument("--dev_csv", default="data/dev.csv")
    parser.add_argument("--ckt_path", required=True)
    parser.add_argument("--model_name", default="gogamza/kobart-base-v2")
    parser.add_argument("--output_dir", default="prediction")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── dev 데이터 로드 ────────────────────────────────────────────────────
    dev_path = Path(args.dev_csv)
    if not dev_path.is_absolute():
        dev_path = _ROOT / dev_path
    dev_df = pd.read_csv(dev_path)
    refs = dev_df["summary"].tolist()
    print(f"[Dev] {len(dev_df)}건")

    # ── KoBART dev 예측 생성 ───────────────────────────────────────────────
    print("\n[KoBART dev 예측 생성]")
    kobart_dev_preds = generate_kobart_dev_preds(
        args.ckt_path, args.model_name, dev_df
    )
    kobart_dev_df = dev_df[["fname"]].copy()
    kobart_dev_df["summary"] = kobart_dev_preds
    kobart_dev_df.to_csv(out_dir / "kobart_single_dev.csv", index=False)

    # ── Qwen dev 예측: 가중치 탐색 위해 Qwen test 예측 재사용
    # (Qwen dev 예측이 없으면 KoBART dev 점수만으로 weight 탐색)
    # → Qwen test 파일에서 동일 데이터 수 기준으로 dev proxy 사용
    qwen_test_df = pd.read_csv(args.qwen)
    print(f"[Qwen test] {len(qwen_test_df)}건")

    # ── 가중치 그리드 (dev KoBART 기준) ───────────────────────────────────
    print("\n" + "=" * 65)
    print("▶ 가중치 그리드 (KoBART dev Combined 기준)")
    print("  * Qwen dev 예측 없이 KoBART dev로 단독 점수 + test용 앙상블 생성")
    print("=" * 65)

    kb_combined = _rouge(kobart_dev_preds, refs)
    print(f"  KoBART single dev Combined = {kb_combined:.4f}")

    # ── test 앙상블 생성 (고정 가중치 후보) ───────────────────────────────
    kobart_test_df = pd.read_csv(args.kobart_single)
    weight_candidates = [0.2, 0.3, 0.4, 0.5]

    ensemble_results = []
    print("\n▶ test 앙상블 CSV 생성 (KoBART single + Qwen)")
    ensemble = WeightedEnsemble()
    for w_kb in weight_candidates:
        w_qw = round(1.0 - w_kb, 1)
        label = f"kobart{w_kb}_qwen{w_qw}"
        merged = ensemble.predict([kobart_test_df, qwen_test_df], weights=[w_kb, w_qw])
        out_path = out_dir / f"ensemble_{label}.csv"
        merged.to_csv(out_path, index=False)
        ensemble_results.append({"label": label, "kobart_w": w_kb, "qwen_w": w_qw,
                                  "path": str(out_path)})
        print(f"  저장: {out_path}")

    # 5-fold KoBART 앙상블도 있으면 추가
    if args.kobart_kfold and Path(args.kobart_kfold).exists():
        kobart_kfold_df = pd.read_csv(args.kobart_kfold)
        for w_kb in weight_candidates:
            w_qw = round(1.0 - w_kb, 1)
            label = f"kfold{w_kb}_qwen{w_qw}"
            merged = ensemble.predict([kobart_kfold_df, qwen_test_df], weights=[w_kb, w_qw])
            out_path = out_dir / f"ensemble_{label}.csv"
            merged.to_csv(out_path, index=False)
            ensemble_results.append({"label": label, "kobart_w": w_kb, "qwen_w": w_qw,
                                      "path": str(out_path)})
            print(f"  저장: {out_path}")

    # ── 요약 저장 ──────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(ensemble_results)
    summary_df.to_csv(out_dir / "cross_ensemble_summary.csv", index=False)

    print(f"\n[완료] KoBART dev Combined = {kb_combined:.4f}")
    print(f"앙상블 후보 {len(ensemble_results)}종 생성:")
    for r in ensemble_results:
        print(f"  {r['label']:30s}  → {r['path']}")
    print(f"\n가중치 스윕 요약: {out_dir}/cross_ensemble_summary.csv")
    print("※ Qwen dev 예측이 있으면 dev ROUGE로 최적 가중치 확정 가능")


if __name__ == "__main__":
    main()
