"""
KoBART → Qwen3 Local Refinement 파이프라인
==========================================
KoBART가 생성한 draft 요약을 Qwen3-14B로 다듬는 후처리 스크립트.
Solar API 없이 로컬에서 실행.

사용법:
    # dev 검증 (kobart_single_dev.csv draft 자동 사용)
    python scripts/kobart_qwen_refine.py --mode dev

    # test 추론 (beam4 단일 모델 draft)
    python scripts/kobart_qwen_refine.py --mode test --draft prediction/kobart_single_beam4.csv

    # k-fold 앙상블 draft 사용
    python scripts/kobart_qwen_refine.py --mode test --draft prediction/kobart_kfold_test.csv

    # LoRA adapter 포함 (선택)
    python scripts/kobart_qwen_refine.py --mode dev \\
        --lora_path LLM/response_only_SFT/outputs/exp_B_r32_a64_lr1e4/lora_adapter

출력:
    prediction/kobart_qwen_refine_{draft_stem}_{mode}.csv
"""

# ── Unsloth 임포트 차단 (혹시 Unsloth 환경에서 실행되는 경우 대비) ────────────
import builtins
_orig_import = builtins.__import__

def _blocking_import(name, *args, **kwargs):
    if name.split(".")[0] in {"unsloth", "unsloth_zoo"}:
        raise ImportError(f"[kobart_qwen_refine] Unsloth import blocked: {name}")
    return _orig_import(name, *args, **kwargs)

builtins.__import__ = _blocking_import
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import gc
import sys
import argparse
from pathlib import Path

import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# 프로젝트 루트를 경로에 추가
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "LLM"))

BASE_MODEL = "unsloth/qwen3-14b-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 2048
MAX_NEW_TOKENS = 192

# ── Refine 프롬프트 ───────────────────────────────────────────────────────────

REFINE_SYSTEM = (
    "당신은 한국어 대화 요약 전문가입니다.\n"
    "주어진 대화와 초안 요약을 보고, 초안의 핵심 내용과 표현을 최대한 유지하면서 자연스럽게 다듬어 주세요.\n\n"
    "규칙:\n"
    "1. 초안 요약의 핵심 키워드·표현을 그대로 유지하세요 (ROUGE 점수 보존이 최우선)\n"
    "2. #Person1#, #Person2# 등 화자 태그를 반드시 포함하세요 (누락된 경우 대화를 보고 복원)\n"
    "3. 문법 오류, 어색한 표현, 화자 태그 공백(#Person 1# → #Person1#)만 수정하세요\n"
    "4. 새로운 정보를 추가하거나 기존 내용을 삭제하지 마세요\n"
    "5. 1~3문장, 마침표로 끝내세요\n"
    "6. 요약문만 출력하세요 (설명·접두사 없이)"
)

REFINE_USER = "[대화]\n{dialogue}\n\n[초안 요약]\n{draft_summary}"


# ── 후처리 ────────────────────────────────────────────────────────────────────

def postprocess(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"#\s*Person\s*(\d+)\s*#", r"#Person\1#", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(요약\s*:\s*|Summary\s*:\s*|다듬은 요약\s*:\s*)", "", text).strip()
    return text if text else "빈 요약"


# ── ROUGE 평가 ────────────────────────────────────────────────────────────────

def compute_rouge(preds: list, golds: list) -> dict:
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
    r1 = scores["rouge-1"]["f"]
    r2 = scores["rouge-2"]["f"]
    rl = scores["rouge-l"]["f"]
    return {"r1": r1, "r2": r2, "rl": rl, "combined": r1 + r2 + rl, "method": method}


def print_rouge(label: str, scores: dict):
    print(
        f"[{label}]  R1={scores['r1']:.4f}  R2={scores['r2']:.4f}  "
        f"RL={scores['rl']:.4f}  Combined={scores['combined']:.4f}  [{scores['method']}]"
    )


# ── 모델 로드 ─────────────────────────────────────────────────────────────────

def load_model(lora_path: str | None):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    if lora_path:
        import json
        cfg = json.load(open(os.path.join(lora_path, "adapter_config.json")))
        base_name = cfg["base_model_name_or_path"]
        print(f"[모델] base: {base_name}  +  adapter: {lora_path}")
    else:
        base_name = BASE_MODEL
        print(f"[모델] base only: {base_name}")

    tokenizer = AutoTokenizer.from_pretrained(lora_path or base_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_name,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    if lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    return model, tokenizer


# ── 배치 추론 ─────────────────────────────────────────────────────────────────

def run_refine(
    model,
    tokenizer,
    dialogues: list,
    drafts: list,
    batch_size: int = 4,
) -> list:
    """dialogue + draft → refined summary (배치 추론)"""
    # 메시지 → 텍스트 변환
    texts = []
    for dialogue, draft in zip(dialogues, drafts):
        messages = [
            {"role": "system", "content": REFINE_SYSTEM},
            {"role": "user",   "content": REFINE_USER.format(
                dialogue=dialogue, draft_summary=draft
            )},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        texts.append(text)

    preds = []
    device = next(model.parameters()).device
    total = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(texts), batch_size), total=total, desc="[Refine]"):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt",
            padding=True, truncation=True, max_length=MAX_SEQ_LENGTH,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        for seq in out:
            raw = tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()
            preds.append(postprocess(raw))

    return preds


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KoBART → Qwen3 Local Refinement")
    parser.add_argument(
        "--mode", choices=["dev", "test"], default="dev",
        help="dev: ROUGE 평가 포함 / test: 제출용 CSV 생성"
    )
    parser.add_argument(
        "--draft", type=str, default=None,
        help=(
            "KoBART draft CSV 경로 (기본값: dev→kobart_single_dev.csv, "
            "test→kobart_single_beam4.csv)"
        ),
    )
    parser.add_argument(
        "--data_dir", type=str,
        default=str(ROOT_DIR / "data"),
        help="train/dev/test CSV가 있는 디렉토리"
    )
    parser.add_argument(
        "--pred_dir", type=str,
        default=str(ROOT_DIR / "prediction"),
        help="결과 저장 디렉토리"
    )
    parser.add_argument(
        "--lora_path", type=str, default=None,
        help="LoRA adapter 경로 (미지정 시 베이스 모델만 사용)"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--n_sample", type=int, default=None,
        help="빠른 검증용 샘플 수 (기본값: 전체)"
    )
    parser.add_argument(
        "--show", type=int, default=5,
        help="비교 출력 샘플 수"
    )
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(exist_ok=True)

    # ── 1. 데이터 로드 ────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    target_csv = data_dir / ("dev.csv" if args.mode == "dev" else "test.csv")
    target_df  = pd.read_csv(target_csv)
    print(f"[데이터] {target_csv.name}: {len(target_df)}개")

    # draft CSV 결정
    if args.draft:
        draft_path = Path(args.draft)
    elif args.mode == "dev":
        draft_path = pred_dir / "kobart_single_dev.csv"
    else:
        draft_path = pred_dir / "kobart_single_beam4.csv"

    draft_df = pd.read_csv(draft_path)
    print(f"[Draft]  {draft_path.name}: {len(draft_df)}개")

    # 열 이름 자동 탐지 (summary / pred_summary 모두 허용)
    summary_col = "summary" if "summary" in draft_df.columns else "pred_summary"
    drafts   = draft_df[summary_col].tolist()
    dialogues = target_df["dialogue"].tolist()

    assert len(drafts) == len(dialogues), (
        f"대화({len(dialogues)}) ≠ draft({len(drafts)}) 개수 불일치"
    )

    # 샘플링 (빠른 검증용)
    if args.n_sample:
        target_df  = target_df.iloc[: args.n_sample]
        dialogues  = dialogues[: args.n_sample]
        drafts     = drafts[: args.n_sample]
        print(f"[샘플링] {args.n_sample}개로 제한")

    # ── 2. 모델 로드 ──────────────────────────────────────────────────────────
    model, tokenizer = load_model(args.lora_path)

    # ── 3. Refine 추론 ────────────────────────────────────────────────────────
    refined = run_refine(model, tokenizer, dialogues, drafts, batch_size=args.batch_size)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── 4. Dev 평가 ───────────────────────────────────────────────────────────
    if args.mode == "dev" and "summary" in target_df.columns:
        golds = target_df["summary"].tolist()

        draft_scores   = compute_rouge(drafts,   golds)
        refined_scores = compute_rouge(refined,  golds)

        print("\n" + "=" * 60)
        print_rouge("KoBART draft", draft_scores)
        print_rouge("Qwen3 refine", refined_scores)
        delta = refined_scores["combined"] - draft_scores["combined"]
        print(f"  → Δ Combined: {delta:+.4f}  ({'개선 ✅' if delta > 0 else '하락 ❌'})")
        print("=" * 60)

        # 샘플 비교 출력
        print(f"\n[샘플 비교 (처음 {args.show}개)]")
        for i in range(min(args.show, len(dialogues))):
            print(f"\n--- [{i}] ---")
            print(f"  대화    : {dialogues[i][:80]}...")
            print(f"  KoBART  : {drafts[i]}")
            print(f"  Refined : {refined[i]}")
            if golds:
                print(f"  Gold    : {golds[i]}")

    # ── 5. 결과 저장 ──────────────────────────────────────────────────────────
    draft_stem = draft_path.stem          # e.g. "kobart_single_beam4"
    adapter_tag = "_lora" if args.lora_path else ""
    sample_tag  = f"_n{args.n_sample}" if args.n_sample else ""
    out_name = f"kobart_qwen_refine_{draft_stem}{adapter_tag}{sample_tag}_{args.mode}.csv"
    out_path = pred_dir / out_name

    if args.mode == "test":
        out_df = pd.DataFrame({"fname": target_df["fname"], "summary": refined})
    else:
        out_df = target_df.copy()
        out_df["draft_summary"]   = drafts
        out_df["refined_summary"] = refined

    out_df.to_csv(out_path, index=False)
    print(f"\n[저장] {out_path}  ({len(out_df)}행)")


if __name__ == "__main__":
    main()
