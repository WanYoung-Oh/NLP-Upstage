"""
KoBART 디코딩 파라미터 그리드 (단계 2-A)

max_new_tokens × length_penalty 조합을 dev에서 비교합니다.
모델은 한 번만 로드하고 여러 설정을 순차 실행합니다.

실행:
    python scripts/kobart_decode_grid.py \
        --ckt_path checkpoints/260324_run_003/epoch06_0.7962 \
        --model_name gogamza/kobart-base-v2
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.metrics import _rouge_baseline
from src.utils.postprocess import batch_postprocess


def load_model(ckt_path: str, model_name: str, special_tokens: list[str]):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from src.utils.device import get_device

    resolved = ckt_path if os.path.isabs(ckt_path) else os.path.join(_ROOT, ckt_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model = AutoModelForSeq2SeqLM.from_pretrained(resolved)
    model.resize_token_embeddings(len(tokenizer))
    device = get_device()
    model.to(device).eval()
    print(f"[모델] {resolved}  device={device}")
    return model, tokenizer, device


def run_beam(model, tokenizer, device, dialogues: list[str],
             num_beams: int, max_new_tokens: int, length_penalty: float,
             no_repeat_ngram_size: int = 3, batch_size: int = 8) -> list[str]:
    from torch.utils.data import DataLoader
    from src.data.preprocess import DatasetForInference

    tok = tokenizer(
        dialogues, return_tensors="pt", padding=True,
        truncation=True, max_length=512,
        add_special_tokens=True, return_token_type_ids=False,
    )
    dataset = DatasetForInference(tok, pd.Series(range(len(dialogues))))
    loader = DataLoader(dataset, batch_size=batch_size)
    label = f"beam{num_beams} mnt{max_new_tokens} lp{length_penalty}"

    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=label, leave=False):
            out = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True,
            )
            preds.extend([tokenizer.decode(ids, skip_special_tokens=True) for ids in out])
    return preds


def rouge(preds: list[str], refs: list[str]) -> dict:
    s = _rouge_baseline(
        [p if p.strip() else "." for p in preds],
        [r if r.strip() else "." for r in refs],
    )
    return {"r1": s["rouge-1"], "r2": s["rouge-2"], "rl": s["rouge-l"],
            "combined": s["rouge-1"] + s["rouge-2"] + s["rouge-l"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckt_path", required=True)
    parser.add_argument("--model_name", default="gogamza/kobart-base-v2")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_csv", default="prediction/kobart_decode_grid.csv")
    args = parser.parse_args()

    special_tokens = [
        "#Person1#", "#Person2#", "#Person3#", "#Person4#",
        "#Person5#", "#Person6#", "#Person7#",
        "#PhoneNumber#", "#Address#", "#PassportNumber#",
        "#DateOfBirth#", "#SSN#", "#CardNumber#", "#CarNumber#", "#Email#",
    ]

    dev_df = pd.read_csv(_ROOT / args.data_dir / "dev.csv")
    dialogues = dev_df["dialogue"].tolist()
    refs = dev_df["summary"].tolist()
    print(f"[Dev] {len(dev_df)}건")

    model, tokenizer, device = load_model(args.ckt_path, args.model_name, special_tokens)
    remove_tokens = ["<s>", "</s>", "<pad>", "<unk>"]

    # ── 그리드 정의 (num_beams, max_new_tokens, length_penalty) ──────────────
    configs = [
        (4, 100, 1.0),   # 현재 기준 → 0.4289
        (4, 150, 1.0),   # max_new_tokens 증가 (95th pct ~110 토큰)
        (4, 200, 1.0),   # max_new_tokens 추가 증가
        (4, 100, 0.8),   # length_penalty 축소 → 짧은 요약 선호
        (4, 100, 1.2),   # length_penalty 확대 → 긴 요약 선호
        (4, 150, 1.2),   # mnt+lp 조합
        (4, 150, 0.8),   # mnt+lp 조합
    ]

    results = []
    print("\n" + "=" * 65)
    for num_beams, mnt, lp in configs:
        label = f"beam{num_beams} mnt={mnt} lp={lp}"
        preds_raw = run_beam(model, tokenizer, device, dialogues,
                             num_beams=num_beams, max_new_tokens=mnt,
                             length_penalty=lp, batch_size=args.batch_size)
        preds_pp = batch_postprocess(preds_raw, remove_tokens)
        s = rouge(preds_pp, refs)
        s["label"] = label
        results.append(s)
        print(f"[{label}]  R1={s['r1']:.4f}  R2={s['r2']:.4f}  "
              f"RL={s['rl']:.4f}  Combined={s['combined']:.4f}")

    out = _ROOT / args.output_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(results)[["label", "r1", "r2", "rl", "combined"]]
    result_df.to_csv(out, index=False)
    print(f"\n결과 저장: {out}")

    best = result_df.loc[result_df["combined"].idxmax()]
    print(f"▶ 최적: [{best['label']}]  Combined={best['combined']:.4f}")
    delta = best["combined"] - 0.4289
    print(f"▶ 기준(beam4 mnt=100 lp=1.0=0.4289) 대비 Δ={delta:+.4f}")


if __name__ == "__main__":
    main()
