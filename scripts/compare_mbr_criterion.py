"""
TTA 후보를 한 번 생성한 뒤 두 가지 MBR 선택 기준을 비교합니다.
  - rouge-l only  (현재 MBRDecoder 기준)
  - combined R1+R2+RL (대회 점수 기준)

실행:
    python scripts/compare_mbr_criterion.py \
        --ckt_path checkpoints/260324_run_004/epoch27_0.6094 \
        --model_name gogamza/kobart-base-v2
"""

from __future__ import annotations
import argparse, os, sys
import pandas as pd
import torch
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rouge import Rouge
from src.data.preprocess import apply_tta
from src.utils.metrics import _rouge_baseline
from src.utils.postprocess import batch_postprocess


def select_mbr(candidates: list[str], rouge_obj: Rouge, criterion: str) -> str:
    """criterion: 'rougeL' | 'combined'"""
    best, best_score = candidates[0], -1.0
    for i, cand in enumerate(candidates):
        others = [c for j, c in enumerate(candidates) if j != i and c.strip()]
        if not others:
            continue
        cand_safe = cand.strip() if cand.strip() else "."
        try:
            s = rouge_obj.get_scores([cand_safe] * len(others), others, avg=True)
            if criterion == "rougeL":
                score = s["rouge-l"]["f"]
            else:
                score = s["rouge-1"]["f"] + s["rouge-2"]["f"] + s["rouge-l"]["f"]
        except Exception:
            score = 0.0
        if score > best_score:
            best_score = score
            best = cand
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckt_path", required=True)
    parser.add_argument("--model_name", default="gogamza/kobart-base-v2")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    args = parser.parse_args()

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    dev_path = os.path.join(_ROOT, args.data_dir, "dev.csv")
    dev_df = pd.read_csv(dev_path)
    dialogues = dev_df["dialogue"].tolist()
    golds = dev_df["summary"].tolist()
    print(f"dev.csv: {len(dev_df)}건")

    # ── 모델 로드 ────────────────────────────────────────────────────────────
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from src.utils.device import get_device

    ckt_path = args.ckt_path if os.path.isabs(args.ckt_path) \
               else os.path.join(_ROOT, args.ckt_path)

    special_tokens = [
        "#Person1#", "#Person2#", "#Person3#", "#Person4#",
        "#Person5#", "#Person6#", "#Person7#",
        "#PhoneNumber#", "#Address#", "#PassportNumber#",
        "#DateOfBirth#", "#SSN#", "#CardNumber#", "#CarNumber#", "#Email#",
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model = AutoModelForSeq2SeqLM.from_pretrained(ckt_path)
    model.resize_token_embeddings(len(tokenizer))
    device = get_device()
    model.to(device)
    model.eval()
    print(f"모델 로드 완료: {ckt_path}")

    # ── TTA 변형 생성 (원본 + 발화 역전) ────────────────────────────────────
    tta_variants = apply_tta(dialogues, n_ways=3)  # list[list[str]]

    # ── 후보 생성 (각 변형에 대해 beam4) ────────────────────────────────────
    rouge_obj = Rouge()
    preds_rougeL, preds_combined = [], []
    remove_tokens = ["<s>", "</s>", "<pad>", "<unk>"]

    for variants in tqdm(tta_variants, desc="TTA inference"):
        cands = []
        for variant in variants:
            tok = tokenizer(
                [variant], return_tensors="pt", padding=True,
                truncation=True, max_length=512,
                add_special_tokens=True, return_token_type_ids=False,
            )
            with torch.no_grad():
                gen = model.generate(
                    input_ids=tok["input_ids"].to(device),
                    attention_mask=tok["attention_mask"].to(device),
                    num_beams=4, max_new_tokens=args.max_new_tokens,
                    no_repeat_ngram_size=3, early_stopping=True,
                )
            cands.append(tokenizer.decode(gen[0], skip_special_tokens=True))

        preds_rougeL.append(select_mbr(cands, rouge_obj, "rougeL"))
        preds_combined.append(select_mbr(cands, rouge_obj, "combined"))

    # ── 후처리 ───────────────────────────────────────────────────────────────
    preds_rougeL_pp  = batch_postprocess(preds_rougeL, remove_tokens)
    preds_combined_pp = batch_postprocess(preds_combined, remove_tokens)

    # ── ROUGE 계산 ───────────────────────────────────────────────────────────
    def report(preds, label):
        s = _rouge_baseline(
            [p if p.strip() else "." for p in preds],
            [g if g.strip() else "." for g in golds],
        )
        r1, r2, rl = s["rouge-1"], s["rouge-2"], s["rouge-l"]
        comb = r1 + r2 + rl
        print(f"[{label}]  R1={r1:.4f}  R2={r2:.4f}  RL={rl:.4f}  Combined={comb:.4f}")
        return {"label": label, "r1": r1, "r2": r2, "rl": rl, "combined": comb}

    print("\n" + "=" * 60)
    r_l   = report(preds_rougeL_pp,   "TTA MBR (rouge-L only) — 현재")
    r_comb = report(preds_combined_pp, "TTA MBR (combined R1+R2+RL) — 제안")

    delta = r_comb["combined"] - r_l["combined"]
    print(f"\n▶ Δcombined = {delta:+.4f}  ({'개선' if delta > 0 else '하락'})")

    # ── 결과 저장 ────────────────────────────────────────────────────────────
    out = os.path.join(_ROOT, "prediction", "mbr_criterion_comparison.csv")
    pd.DataFrame([r_l, r_comb]).to_csv(out, index=False)
    print(f"결과 저장: {out}")

    # 샘플 차이 확인 (두 기준이 다른 선택을 한 경우)
    diffs = [(i, preds_rougeL_pp[i], preds_combined_pp[i])
             for i in range(len(preds_rougeL_pp))
             if preds_rougeL_pp[i] != preds_combined_pp[i]]
    print(f"\n두 기준이 다른 선택을 한 샘플 수: {len(diffs)} / {len(dialogues)}")
    for i, a, b in diffs[:3]:
        print(f"\n  [sample {i}]")
        print(f"    rouge-L:   {a}")
        print(f"    combined:  {b}")
        print(f"    gold:      {golds[i]}")


if __name__ == "__main__":
    main()
