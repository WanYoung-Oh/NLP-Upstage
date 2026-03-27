"""
Qwen3 MBR → Solar API Refinement 파이프라인
============================================
Qwen3 MBR로 생성한 draft 요약을 Solar API로 정제.

사용법:
    # dev 50개 검증 (먼저 실행)
    python scripts/qwen_solar_refine.py --mode dev --n_sample 50

    # dev 전체
    python scripts/qwen_solar_refine.py --mode dev

    # test 전체
    python scripts/qwen_solar_refine.py --mode test

    # draft CSV 직접 지정
    python scripts/qwen_solar_refine.py --mode test \\
        --draft prediction/qwen_test_mbr_top5_best.csv

    # 재시도 (중단된 경우 --resume)
    python scripts/qwen_solar_refine.py --mode dev --resume

출력:
    prediction/qwen_solar_refine_{draft_stem}_{mode}.csv
"""

import os
import re
import sys
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import requests
from tqdm.auto import tqdm

# ── 프로젝트 루트 경로 설정 ───────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "LLM"))

# ── .env 로드 ─────────────────────────────────────────────────────────────────
def load_env(env_path: Path) -> dict:
    env = {}
    if not env_path.exists():
        return env
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip()
    return env

ENV = load_env(ROOT_DIR / ".env")
SOLAR_API_URL = "https://api.upstage.ai/v1/chat/completions"


# ── ROUGE 평가 ────────────────────────────────────────────────────────────────
def compute_rouge(preds: List[str], golds: List[str]) -> dict:
    from rouge import Rouge
    rouge = Rouge()
    try:
        from prompts.mecab_ko import get_mecab
        m = get_mecab()
        preds_t = [" ".join(m.morphs(p)) if p.strip() else "빈요약" for p in preds]
        golds_t = [" ".join(m.morphs(g)) if g.strip() else "빈요약" for g in golds]
        method = "mecab"
    except Exception:
        preds_t = [p if p.strip() else "빈요약" for p in preds]
        golds_t = [g if g.strip() else "빈요약" for g in golds]
        method = "whitespace"
    scores = rouge.get_scores(preds_t, golds_t, avg=True)
    r1, r2, rl = scores["rouge-1"]["f"], scores["rouge-2"]["f"], scores["rouge-l"]["f"]
    return {"r1": r1, "r2": r2, "rl": rl, "combined": r1 + r2 + rl, "method": method}


def print_rouge(label: str, s: dict):
    print(
        f"[{label}]  R1={s['r1']:.4f}  R2={s['r2']:.4f}  "
        f"RL={s['rl']:.4f}  Combined={s['combined']:.4f}  [{s['method']}]"
    )


# ── Solar API ─────────────────────────────────────────────────────────────────

REFINE_SYSTEM = (
    "당신은 한국어 대화 요약 전문가입니다.\n"
    "주어진 대화와 초안 요약을 보고, 초안의 표현과 내용을 최대한 유지하면서 자연스럽게 다듬어 주세요.\n\n"
    "규칙:\n"
    "1. 초안의 핵심 표현·키워드를 반드시 유지하세요 (ROUGE 점수 보존 최우선)\n"
    "2. #Person1#, #Person2# 등 화자 태그를 반드시 사용하세요\n"
    "3. 어색한 조사·문법 오류만 수정하고, 내용 추가나 삭제는 금지\n"
    "4. 1~3문장, 마침표로 끝내세요\n"
    "5. 요약문만 출력하세요 (설명·접두사 없이)"
)


def build_fewshot_messages(
    dialogue: str,
    draft: str,
    examples: List[Dict],  # [{"dialogue": ..., "draft": ..., "gold": ...}]
) -> List[Dict]:
    """Few-shot refine 메시지 구성"""
    messages = [{"role": "system", "content": REFINE_SYSTEM}]
    for ex in examples:
        user_msg = f"[대화]\n{ex['dialogue']}\n\n[초안 요약]\n{ex['draft']}"
        messages.append({"role": "user",      "content": user_msg})
        messages.append({"role": "assistant", "content": ex["gold"]})
    messages.append({
        "role": "user",
        "content": f"[대화]\n{dialogue}\n\n[초안 요약]\n{draft}",
    })
    return messages


def call_solar(
    messages: List[Dict],
    api_key: str,
    model: str = "solar-pro",
    temperature: float = 0.2,
    max_tokens: int = 256,
    max_retries: int = 4,
) -> Optional[str]:
    """Solar API 호출 (지수 백오프 재시도)"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(SOLAR_API_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except requests.HTTPError as e:
            if resp.status_code == 429:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"  [Rate limit] {wait:.1f}s 대기...")
                time.sleep(wait)
            else:
                print(f"  [HTTP {resp.status_code}] {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2)
        except requests.RequestException as e:
            print(f"  [Request error] {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2)
    return None


def postprocess(text: str) -> str:
    text = re.sub(r"#\s*Person\s*(\d+)\s*#", r"#Person\1#", text)
    text = re.sub(r"^(요약\s*:\s*|Summary\s*:\s*)", "", text)
    return re.sub(r"\s+", " ", text).strip()


# ── Few-shot 예시 샘플링 ───────────────────────────────────────────────────────

def sample_fewshot(
    train_df: pd.DataFrame,
    qwen_dev_df: Optional[pd.DataFrame],
    n: int = 3,
    seed: int = 42,
) -> List[Dict]:
    """
    Few-shot 예시 구성.
    qwen_dev_df가 있으면 Qwen3 dev 예측 → Gold 쌍 사용 (가장 사실적).
    없으면 train Gold → Gold (draft = gold, 스타일 예시용).
    """
    random.seed(seed)

    if qwen_dev_df is not None and len(qwen_dev_df) >= n:
        # dev Qwen3 예측과 dev Gold 매칭 (fname 기준)
        dev_csv = ROOT_DIR / "data" / "dev.csv"
        dev_df  = pd.read_csv(dev_csv)
        merged  = pd.merge(
            dev_df[["fname", "dialogue", "summary"]],
            qwen_dev_df[["fname", "summary"]].rename(columns={"summary": "qwen_summary"}),
            on="fname",
        )
        sampled = merged.sample(n=min(n, len(merged)), random_state=seed)
        return [
            {"dialogue": r["dialogue"], "draft": r["qwen_summary"], "gold": r["summary"]}
            for _, r in sampled.iterrows()
        ]

    # fallback: train Gold (draft = gold)
    sampled = train_df.sample(n=min(n, len(train_df)), random_state=seed)
    return [
        {"dialogue": r["dialogue"], "draft": r["summary"], "gold": r["summary"]}
        for _, r in sampled.iterrows()
    ]


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Qwen3 MBR → Solar Refinement")
    parser.add_argument("--mode", choices=["dev", "test"], default="dev")
    parser.add_argument(
        "--draft", type=str, default=None,
        help=(
            "Qwen3 draft CSV 경로 (기본값: dev→qwen_dev_mbr.csv, "
            "test→qwen_test_mbr_top5_best.csv)"
        ),
    )
    parser.add_argument("--n_sample", type=int, default=None,
                        help="검증용 샘플 수 (기본값: 전체)")
    parser.add_argument("--n_fewshot", type=int, default=3,
                        help="Few-shot 예시 수 (기본값: 3)")
    parser.add_argument("--delay",  type=float, default=0.5,
                        help="API 호출 간격(초)")
    parser.add_argument("--model",  type=str, default="solar-pro")
    parser.add_argument("--resume", action="store_true",
                        help="중단된 경우 기존 결과 이어서 실행")
    parser.add_argument("--api_key", type=str, default=None,
                        help="Upstage API key (기본값: .env의 UPSTAGE_API_KEY)")
    parser.add_argument("--show", type=int, default=5,
                        help="비교 출력 샘플 수")
    args = parser.parse_args()

    # ── API 키 ────────────────────────────────────────────────────────────────
    api_key = args.api_key or ENV.get("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError(".env에 UPSTAGE_API_KEY가 없습니다. --api_key로 지정하세요.")
    print(f"[API] 키 로드됨 (앞 8자: {api_key[:8]}...)")

    pred_dir = ROOT_DIR / "prediction"
    pred_dir.mkdir(exist_ok=True)
    data_dir = ROOT_DIR / "data"

    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    target_csv = data_dir / ("dev.csv" if args.mode == "dev" else "test.csv")
    target_df  = pd.read_csv(target_csv)
    print(f"[데이터] {target_csv.name}: {len(target_df)}개")

    if args.draft:
        draft_path = Path(args.draft)
    elif args.mode == "dev":
        draft_path = pred_dir / "qwen_dev_mbr.csv"
    else:
        draft_path = pred_dir / "qwen_test_mbr_top5_best.csv"

    draft_df = pd.read_csv(draft_path)
    print(f"[Draft]  {draft_path.name}: {len(draft_df)}개")

    # draft 열 이름 자동 탐지
    draft_col = "summary" if "summary" in draft_df.columns else "pred_summary"
    drafts    = draft_df[draft_col].tolist()
    dialogues = target_df["dialogue"].tolist()
    fnames    = target_df["fname"].tolist()

    assert len(drafts) == len(dialogues), (
        f"대화({len(dialogues)}) ≠ draft({len(drafts)}) 개수 불일치"
    )

    # 샘플링
    if args.n_sample:
        target_df = target_df.iloc[: args.n_sample]
        dialogues = dialogues[: args.n_sample]
        drafts    = drafts[: args.n_sample]
        fnames    = fnames[: args.n_sample]
        print(f"[샘플링] {args.n_sample}개로 제한")

    # ── 출력 파일명 결정 ──────────────────────────────────────────────────────
    sample_tag = f"_n{args.n_sample}" if args.n_sample else ""
    out_name   = f"qwen_solar_refine_{draft_path.stem}{sample_tag}_{args.mode}.csv"
    out_path   = pred_dir / out_name

    # ── resume: 이미 처리된 결과 로드 ─────────────────────────────────────────
    done_map: Dict[str, str] = {}  # fname → refined_summary
    if args.resume and out_path.exists():
        prev_df = pd.read_csv(out_path)
        if "refined_summary" in prev_df.columns:
            done_map = dict(zip(prev_df["fname"], prev_df["refined_summary"]))
        elif "summary" in prev_df.columns:
            done_map = dict(zip(prev_df["fname"], prev_df["summary"]))
        print(f"[Resume] 기존 결과 {len(done_map)}개 로드")

    # ── Few-shot 예시 샘플링 ──────────────────────────────────────────────────
    train_df   = pd.read_csv(data_dir / "train.csv")
    qwen_dev_df = pd.read_csv(pred_dir / "qwen_dev_mbr.csv") if (
        pred_dir / "qwen_dev_mbr.csv"
    ).exists() else None

    fewshot_examples = sample_fewshot(train_df, qwen_dev_df, n=args.n_fewshot)
    print(f"\n[Few-shot] {len(fewshot_examples)}개 예시 구성")
    for i, ex in enumerate(fewshot_examples):
        print(f"  예시 {i+1}: {ex['draft'][:60]}... → {ex['gold'][:60]}...")

    # ── Solar Refine 루프 ─────────────────────────────────────────────────────
    refined: List[str] = []
    skip_count = 0
    fail_count = 0

    print(f"\n[Solar Refine 시작]  총 {len(dialogues)}개  모델: {args.model}")
    print("=" * 60)

    for i, (fname, dialogue, draft) in enumerate(
        tqdm(zip(fnames, dialogues, drafts), total=len(dialogues), desc="Solar")
    ):
        # resume: 이미 처리된 항목 스킵
        if fname in done_map:
            refined.append(done_map[fname])
            skip_count += 1
            continue

        messages = build_fewshot_messages(dialogue, draft, fewshot_examples)
        result   = call_solar(messages, api_key, model=args.model)

        if result:
            refined.append(postprocess(result))
        else:
            # API 실패 → draft 원본 유지
            print(f"  [{i}] {fname}: API 실패 → draft 유지")
            refined.append(draft)
            fail_count += 1

        # 중간 저장 (50개마다)
        if (i + 1) % 50 == 0:
            _save_progress(out_path, fnames[:i+1], drafts[:i+1], refined, args.mode, target_df)
            print(f"  → 중간 저장: {i+1}개 완료")

        if args.delay > 0 and i < len(dialogues) - 1:
            time.sleep(args.delay)

    print(f"\n[완료]  성공: {len(refined)-fail_count}  실패(draft유지): {fail_count}  스킵: {skip_count}")

    # ── Dev 평가 ──────────────────────────────────────────────────────────────
    if args.mode == "dev" and "summary" in target_df.columns:
        golds = target_df["summary"].tolist()

        draft_scores   = compute_rouge(drafts,   golds)
        refined_scores = compute_rouge(refined,  golds)

        print("\n" + "=" * 60)
        print_rouge("Qwen3 MBR draft", draft_scores)
        print_rouge("Solar refined  ", refined_scores)
        delta = refined_scores["combined"] - draft_scores["combined"]
        print(f"  → Δ Combined: {delta:+.4f}  ({'개선 ✅' if delta > 0 else '하락 ❌'})")
        print("=" * 60)

        # 샘플 비교
        print(f"\n[샘플 비교 (처음 {args.show}개)]")
        for i in range(min(args.show, len(dialogues))):
            print(f"\n--- [{i}] {fnames[i]} ---")
            print(f"  대화    : {dialogues[i][:80]}...")
            print(f"  Draft   : {drafts[i]}")
            print(f"  Refined : {refined[i]}")
            print(f"  Gold    : {golds[i]}")

    # ── 최종 저장 ─────────────────────────────────────────────────────────────
    _save_progress(out_path, fnames, drafts, refined, args.mode, target_df)
    print(f"\n[저장] {out_path}  ({len(refined)}행)")


def _save_progress(
    out_path: Path,
    fnames: List[str],
    drafts: List[str],
    refined: List[str],
    mode: str,
    target_df: pd.DataFrame,
):
    """중간·최종 저장 (dev는 draft 컬럼 포함, test는 제출 형식)"""
    n = len(refined)
    if mode == "test":
        df = pd.DataFrame({"fname": fnames[:n], "summary": refined})
    else:
        df = pd.DataFrame({
            "fname":            fnames[:n],
            "draft_summary":    drafts[:n],
            "refined_summary":  refined,
        })
        # gold 컬럼 추가 (있는 경우)
        if "summary" in target_df.columns:
            df["gold_summary"] = target_df["summary"].iloc[:n].values
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
