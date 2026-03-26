"""
Test 추론 스크립트 — A/B 비교 후 최고 체크포인트로 test.csv 추론

핵심: 모델 추론은 inference_worker.py를 별도 subprocess로 실행.
      → 현재 프로세스에 Unsloth가 임포트돼 있어도 worker는 완전히 격리됨.

사용법:
    cd /data/ephemeral/home/NLP/LLM/response_only_SFT
    python run_test_inference.py
    python run_test_inference.py --exp exp_A_r64_a128_lr2e4
"""

import os
import sys
import json
import argparse
import subprocess
from typing import List

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# 경로
# ──────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_DIR    = os.path.dirname(SCRIPT_DIR)
ROOT_DIR   = os.path.dirname(LLM_DIR)

for p in [LLM_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

DATA_PATH  = os.path.join(SCRIPT_DIR, "data")
PRED_PATH  = os.path.join(SCRIPT_DIR, "prediction")
WORKER     = os.path.join(SCRIPT_DIR, "inference_worker.py")
PYTHON     = sys.executable


# ──────────────────────────────────────────────────────────────────────────────
# subprocess 추론 호출
# ──────────────────────────────────────────────────────────────────────────────
def call_worker(lora_path: str, input_csv: str, output_csv: str,
                mode: str = "test", batch_size: int = 4) -> bool:
    """inference_worker.py를 별도 프로세스로 실행. 성공 여부 반환."""
    cmd = [
        PYTHON, WORKER,
        "--lora_path",  lora_path,
        "--input_csv",  input_csv,
        "--output_csv", output_csv,
        "--mode",       mode,
        "--batch_size", str(batch_size),
    ]
    print(f"[subprocess] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    return result.returncode == 0


# ──────────────────────────────────────────────────────────────────────────────
# ROUGE 평가
# ──────────────────────────────────────────────────────────────────────────────
def compute_rouge_combined(preds: List[str], golds: List[str]) -> dict:
    from rouge import Rouge
    rouge = Rouge()
    try:
        from prompts.mecab_ko import get_mecab
        m = get_mecab()
        preds_m = [" ".join(m.morphs(p)) if p.strip() else "빈요약" for p in preds]
        golds_m = [" ".join(m.morphs(g)) if g.strip() else "빈요약" for g in golds]
        scores = rouge.get_scores(preds_m, golds_m, avg=True)
        method = "mecab"
    except Exception:
        preds_s = [p if p.strip() else "빈요약" for p in preds]
        golds_s = [g if g.strip() else "빈요약" for g in golds]
        scores = rouge.get_scores(preds_s, golds_s, avg=True)
        method = "whitespace"
    r1 = scores["rouge-1"]["f"]
    r2 = scores["rouge-2"]["f"]
    rl = scores["rouge-l"]["f"]
    return {"r1": r1, "r2": r2, "rl": rl, "combined": r1 + r2 + rl, "method": method}


# ──────────────────────────────────────────────────────────────────────────────
# dev 추론 (누락 시 worker 호출)
# ──────────────────────────────────────────────────────────────────────────────
def run_dev_inference_if_missing(exp_name: str, batch_size: int = 1) -> bool:
    pred_file = os.path.join(PRED_PATH, f"dev_{exp_name}_qa_style.csv")
    if os.path.exists(pred_file):
        print(f"  [{exp_name}] dev 추론 파일 존재 → 스킵")
        return True

    lora_path = os.path.join(SCRIPT_DIR, "outputs", exp_name, "lora_adapter")
    if not os.path.exists(os.path.join(lora_path, "adapter_model.safetensors")):
        print(f"  [{exp_name}] adapter 없음 → dev 추론 불가")
        return False

    print(f"\n  [{exp_name}] dev 추론 시작 (subprocess, batch_size={batch_size})...")
    dev_csv = os.path.join(DATA_PATH, "dev.csv")
    ok = call_worker(lora_path, dev_csv, pred_file, mode="dev", batch_size=batch_size)
    if ok:
        print(f"  [{exp_name}] dev 추론 완료: {pred_file}")
    else:
        print(f"  [{exp_name}] dev 추론 실패")
    return ok


# ──────────────────────────────────────────────────────────────────────────────
# A/B 비교 → 최고 실험 결정
# ──────────────────────────────────────────────────────────────────────────────
def pick_best_experiment(exp_names: List[str], batch_size: int) -> str:
    for name in exp_names:
        run_dev_inference_if_missing(name, batch_size=batch_size)

    dev_df = pd.read_csv(os.path.join(DATA_PATH, "dev.csv"))
    results = []
    for name in exp_names:
        pred_file = os.path.join(PRED_PATH, f"dev_{name}_qa_style.csv")
        if not os.path.exists(pred_file):
            print(f"[경고] {name} dev 결과 없음 → 비교 제외")
            continue
        preds = pd.read_csv(pred_file)["pred_summary"].tolist()
        golds = dev_df["summary"].tolist()
        s = compute_rouge_combined(preds, golds)
        results.append({"name": name, **s})
        print(f"  {name}: Combined={s['combined']:.4f}  "
              f"(R1={s['r1']:.4f}, R2={s['r2']:.4f}, RL={s['rl']:.4f}) [{s['method']}]")

    if not results:
        raise RuntimeError("비교할 dev 추론 결과가 없습니다.")

    results.sort(key=lambda x: x["combined"], reverse=True)
    best = results[0]
    print(f"\n→ 최고 실험: {best['name']}  Combined={best['combined']:.4f}")
    return best["name"]


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default=None,
                        help="실험 이름 지정 (미지정 시 A/B 자동 비교)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="배치 크기 (기본 4, OOM 시 2 또는 1)")
    args = parser.parse_args()

    os.makedirs(PRED_PATH, exist_ok=True)

    # ── 1. 최고 실험 결정 ────────────────────────────────────────────────────
    available_exps = ["exp_A_r64_a128_lr2e4", "exp_B_r32_a64_lr1e4"]

    if args.exp:
        best_exp = args.exp
        print(f"지정된 실험: {best_exp}")
    else:
        print("=" * 60)
        print("A/B 실험 dev ROUGE 비교")
        print("=" * 60)
        best_exp = pick_best_experiment(available_exps, batch_size=args.batch_size)

    # ── 2. adapter 확인 ──────────────────────────────────────────────────────
    lora_path = os.path.join(SCRIPT_DIR, "outputs", best_exp, "lora_adapter")
    if not os.path.exists(os.path.join(lora_path, "adapter_model.safetensors")):
        raise FileNotFoundError(f"LoRA adapter 없음: {lora_path}")

    # ── 3. test 추론 중복 확인 ───────────────────────────────────────────────
    out_csv = os.path.join(PRED_PATH, f"test_{best_exp}_qa_style.csv")
    if os.path.exists(out_csv):
        print(f"test 추론 결과 이미 존재: {out_csv}")
        print("재실행하려면 해당 파일을 삭제 후 재시도하세요.")
        return

    # ── 4. test 추론 (subprocess) ────────────────────────────────────────────
    test_csv = os.path.join(DATA_PATH, "test.csv")
    test_df  = pd.read_csv(test_csv)
    print(f"\n[test 추론 시작]  {best_exp}  샘플: {len(test_df):,}개  batch_size={args.batch_size}")
    print("=" * 60)

    ok = call_worker(lora_path, test_csv, out_csv, mode="test", batch_size=args.batch_size)

    if ok:
        print(f"\n완료! 제출 파일: {out_csv}")
    else:
        print("\n[오류] batch_size=4 실패 → batch_size=1로 재시도...")
        ok = call_worker(lora_path, test_csv, out_csv, mode="test", batch_size=1)
        if ok:
            print(f"\n완료 (batch_size=1)! 제출 파일: {out_csv}")
        else:
            print("\n[오류] test 추론 최종 실패. 로그를 확인하세요.")
            sys.exit(1)


if __name__ == "__main__":
    main()
