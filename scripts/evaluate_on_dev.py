"""
dev.csv에 대해 추론을 실행하고 다음을 비교하는 평가 스크립트.

1. 후처리 전/후 ROUGE 비교  (PRD Phase 5: "후처리 전/후 Dev ROUGE 변화 확인")
2. beam4 vs beam8 vs MBR ROUGE 비교 (PRD Phase 5: "beam4 vs beam8 vs MBR 성능 비교")
3. TTA(2-way) ROUGE 측정  (PRD Phase 5: "8-way TTA → ROUGE 투표 방식 검증")

실행 예시:
    # 기본 (best checkpoint 자동 탐색)
    python scripts/evaluate_on_dev.py

    # 체크포인트 직접 지정
    python scripts/evaluate_on_dev.py --ckt_path checkpoints/260314_run_001/epoch07_0.7566

    # beam8 + TTA 포함 전체 비교
    python scripts/evaluate_on_dev.py --ckt_path <path> --run_all
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

import pandas as pd
import torch
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.preprocess import Preprocess, apply_tta, reverse_utterances
from src.utils.device import get_device
from src.utils.metrics import _rouge_baseline, evaluate_multi_ref
from src.utils.postprocess import batch_postprocess, batch_postprocess_with_flags


def find_best_checkpoint(checkpoints_root: str = "checkpoints") -> str | None:
    """checkpoints/{run_id}/epoch##_score 구조에서 best 체크포인트 자동 탐색."""
    run_pat = re.compile(r"^\d{6}_run_\d+$")
    epoch_pat = re.compile(r"^epoch\d+_([\d.]+)$")
    best_path, best_score = None, -1.0
    root = os.path.join(_ROOT, checkpoints_root)
    if not os.path.isdir(root):
        return None
    for run_name in os.listdir(root):
        run_dir = os.path.join(root, run_name)
        if not (run_pat.match(run_name) and os.path.isdir(run_dir)):
            continue
        for name in os.listdir(run_dir):
            m = epoch_pat.match(name)
            if m:
                score = float(m.group(1))
                if score > best_score:
                    best_score = score
                    best_path = os.path.join(run_dir, name)
    return best_path


def load_model_tokenizer(ckt_path: str, model_name: str, special_tokens: list[str]):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    # 상대 경로를 절대 경로로 변환
    if not os.path.isabs(ckt_path):
        ckt_path = os.path.join(_ROOT, ckt_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model = AutoModelForSeq2SeqLM.from_pretrained(ckt_path)
    model.resize_token_embeddings(len(tokenizer))
    device = get_device()
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _generate_summaries(
    model, tokenizer, device, dialogues: list[str],
    num_beams: int = 4, do_sample: bool = False,
    max_new_tokens: int = 100, length_penalty: float = 1.0,
    n_samples: int = 10,
    encoder_max_len: int = 512,
    batch_size: int = 8,
) -> list[str]:
    """dialogues 리스트에 대해 배치 추론 후 요약 텍스트 리스트 반환."""
    from torch.utils.data import DataLoader
    from src.data.preprocess import DatasetForInference

    tok_kw = dict(
        return_tensors="pt", padding=True, add_special_tokens=True,
        truncation=True, max_length=encoder_max_len, return_token_type_ids=False,
    )
    tokenized = tokenizer(dialogues, **tok_kw)
    dataset = DatasetForInference(tokenized, pd.Series(range(len(dialogues))))
    loader = DataLoader(dataset, batch_size=batch_size)

    summaries: list[str] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"beam={num_beams} sample={do_sample}", leave=False):
            gen_kw = dict(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=max_new_tokens,
            )
            if do_sample:
                gen_kw.update(do_sample=True, temperature=0.9, top_p=0.95,
                              num_return_sequences=n_samples)
                generated = model.generate(**gen_kw)
                bs = batch["input_ids"].size(0)
                for i in range(bs):
                    candidates = [tokenizer.decode(generated[i * n_samples + j],
                                                   skip_special_tokens=True)
                                  for j in range(n_samples)]
                    from inference import _mbr_select  # noqa: E402
                    summaries.append(_mbr_select(candidates))
            else:
                gen_kw.update(num_beams=num_beams, length_penalty=length_penalty,
                              no_repeat_ngram_size=3, early_stopping=True)
                generated = model.generate(**gen_kw)
                summaries.extend([tokenizer.decode(ids, skip_special_tokens=True)
                                   for ids in generated])
    return summaries


def compute_rouge_report(preds: list[str], golds: list[str], label: str) -> dict:
    scores = _rouge_baseline(
        [p if p.strip() else "." for p in preds],
        [g if g.strip() else "." for g in golds],
    )
    r1, r2, rl = scores["rouge-1"], scores["rouge-2"], scores["rouge-l"]
    combined = r1 + r2 + rl
    print(f"\n[{label}]")
    print(f"  ROUGE-1: {r1:.4f}  ROUGE-2: {r2:.4f}  ROUGE-L: {rl:.4f}  Combined: {combined:.4f}")
    return {"label": label, "r1": r1, "r2": r2, "rl": rl, "combined": combined}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckt_path", default=None, help="체크포인트 경로 (미지정 시 자동 탐색)")
    parser.add_argument("--model_name", default="digit82/kobart-summarization",
                        help="베이스 모델 이름 (토크나이저 로드용)")
    parser.add_argument("--data_dir", default="data", help="dev.csv 위치")
    parser.add_argument("--run_all", action="store_true",
                        help="beam4/beam8/MBR/TTA 전체 비교 실행")
    parser.add_argument("--n_tta_ways", type=int, default=2, help="TTA 변형 수")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_csv", default="prediction/dev_eval_results.csv",
                        help="결과 저장 CSV 경로")
    args = parser.parse_args()

    # ── 체크포인트 확정 ─────────────────────────────────────────────────────
    ckt_path = args.ckt_path or find_best_checkpoint()
    if ckt_path is None:
        print("[오류] 체크포인트를 찾을 수 없습니다. --ckt_path 를 지정하세요.")
        sys.exit(1)
    print(f"[Info] 체크포인트: {ckt_path}")

    # ── dev.csv 로드 ─────────────────────────────────────────────────────────
    dev_path = os.path.join(_ROOT, args.data_dir, "dev.csv")
    dev_df = pd.read_csv(dev_path)
    dialogues = dev_df["dialogue"].tolist()
    golds = dev_df["summary"].tolist()
    print(f"[Info] dev.csv: {len(dev_df)}건 로드")

    # ── 특수 토큰 (config.yaml에서 읽거나 직접 정의) ──────────────────────────
    special_tokens = [
        "#Person1#", "#Person2#", "#Person3#",
        "#Person4#", "#Person5#", "#Person6#", "#Person7#",
        "#PhoneNumber#", "#Address#", "#PassportNumber#",
        "#DateOfBirth#", "#SSN#", "#CardNumber#", "#CarNumber#", "#Email#",
    ]

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    print(f"[Info] 모델 로드: {args.model_name}")
    model, tokenizer, device = load_model_tokenizer(
        ckt_path, args.model_name, special_tokens
    )
    remove_tokens = ["<s>", "</s>", "<pad>", "<unk>"]

    results = []

    # ════════════════════════════════════════════════════════════════════════
    # A. beam4 기본 추론
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("A. beam4 기본 추론")
    preds_raw = _generate_summaries(
        model, tokenizer, device, dialogues,
        num_beams=4, max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    # 후처리 전
    results.append(compute_rouge_report(preds_raw, golds, "beam4 (후처리 전)"))

    # 후처리 후  ← PRD: "후처리 전/후 Dev ROUGE 변화 확인"
    preds_pp, flags = batch_postprocess_with_flags(preds_raw, remove_tokens)
    results.append(compute_rouge_report(preds_pp, golds, "beam4 (후처리 후)"))
    short = sum(flags)
    if short:
        print(f"  ※ {short}/{len(preds_pp)}개 요약이 최소 길이 미달")

    # 후처리 전/후 ROUGE 변화 출력
    pre = results[-2]["combined"]
    post = results[-1]["combined"]
    print(f"\n  ▶ 후처리 효과: combined {pre:.4f} → {post:.4f} (Δ{post - pre:+.4f})")

    # ════════════════════════════════════════════════════════════════════════
    # B. beam8 (--run_all 시)  ← PRD: "beam4 vs beam8 vs MBR 성능 비교"
    # ════════════════════════════════════════════════════════════════════════
    if args.run_all:
        print("\n" + "=" * 60)
        print("B. beam8 추론")
        preds_b8 = _generate_summaries(
            model, tokenizer, device, dialogues,
            num_beams=8, length_penalty=1.2,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        preds_b8_pp = batch_postprocess(preds_b8, remove_tokens)
        results.append(compute_rouge_report(preds_b8_pp, golds, "beam8 (후처리 후)"))

        # ── MBR decoding ───────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("C. MBR decoding (n_samples=10)")
        from src.ensemble import MBRDecoder
        mbr_decoder = MBRDecoder()

        preds_mbr_raw: list[str] = []
        for i, dlg in enumerate(tqdm(dialogues, desc="MBR", leave=False)):
            with torch.no_grad():
                tokenized = tokenizer(
                    [dlg], return_tensors="pt", padding=True,
                    truncation=True, max_length=512,
                    add_special_tokens=True, return_token_type_ids=False,
                )
                gen = model.generate(
                    input_ids=tokenized["input_ids"].to(device),
                    attention_mask=tokenized["attention_mask"].to(device),
                    do_sample=True, temperature=0.9, top_p=0.95,
                    num_beams=1,
                    num_return_sequences=10,
                    max_new_tokens=args.max_new_tokens,
                )
                candidates = [tokenizer.decode(gen[j], skip_special_tokens=True)
                              for j in range(10)]
                preds_mbr_raw.append(mbr_decoder.decode(candidates))

        preds_mbr_pp = batch_postprocess(preds_mbr_raw, remove_tokens)
        results.append(compute_rouge_report(preds_mbr_pp, golds, "MBR (후처리 후)"))

        # ── 비교 요약 ──────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("▶ beam4 vs beam8 vs MBR 비교 요약")
        for r in results:
            if r["label"] in ("beam4 (후처리 후)", "beam8 (후처리 후)", "MBR (후처리 후)"):
                print(f"  {r['label']:25s} | combined={r['combined']:.4f} "
                      f"| R1={r['r1']:.4f} R2={r['r2']:.4f} RL={r['rl']:.4f}")

        # ── TTA (2-way) ────────────────────────────────────────────────────
        # PRD: "8-way TTA → ROUGE 투표 방식 검증" (여기서는 n_tta_ways 설정으로 제어)
        print("\n" + "=" * 60)
        print(f"D. TTA ({args.n_tta_ways}-way: 원본 + 발화 역전)")
        tta_variants = apply_tta(dialogues, n_ways=args.n_tta_ways)
        mbr_dec = MBRDecoder()
        preds_tta: list[str] = []

        for i, variants in enumerate(tqdm(tta_variants, desc="TTA", leave=False)):
            cands = []
            for variant in variants:
                with torch.no_grad():
                    tok = tokenizer(
                        [variant], return_tensors="pt", padding=True,
                        truncation=True, max_length=512,
                        add_special_tokens=True, return_token_type_ids=False,
                    )
                    gen = model.generate(
                        input_ids=tok["input_ids"].to(device),
                        attention_mask=tok["attention_mask"].to(device),
                        num_beams=4, max_new_tokens=args.max_new_tokens,
                        no_repeat_ngram_size=3, early_stopping=True,
                    )
                    cands.append(tokenizer.decode(gen[0], skip_special_tokens=True))
            preds_tta.append(mbr_dec.decode(cands))

        preds_tta_pp = batch_postprocess(preds_tta, remove_tokens)
        results.append(compute_rouge_report(preds_tta_pp, golds,
                                            f"TTA {args.n_tta_ways}-way (후처리 후)"))

    # ════════════════════════════════════════════════════════════════════════
    # 결과 저장
    # ════════════════════════════════════════════════════════════════════════
    out_path = os.path.join(_ROOT, args.output_csv)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path, index=False)
    print(f"\n[완료] 평가 결과 저장: {out_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
