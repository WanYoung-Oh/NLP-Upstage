#!/usr/bin/env python3
"""
멀티 체크포인트 MBR 앙상블 — conf/ensemble_mode*.yaml + conf/checkpoints.yaml

토크나이저: 체크포인트마다 해당 어댑터(학습 산출물) 폴더에 tokenizer.save_pretrained로
저장된 것만 로드합니다. 베이스 모델 이름으로 토크나이저를 불러오지 않습니다.

사용 예:
  cd /data/ephemeral/home/NLP/LLM
  python run_ensemble.py --config conf/ensemble_mode1_quick.yaml \\
      --test_file response_only_SFT/data/test.csv \\
      --output_file ../prediction/ensemble_mode1.csv
"""

from __future__ import annotations

import argparse
import gc
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import torch
import yaml

# Unsloth 먼저
import unsloth  # noqa: F401
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# 프로젝트 루트
LLM_DIR = Path(__file__).resolve().parent
ROOT_DIR = LLM_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(LLM_DIR))

from prompts.mbr_decoding import (
    apply_mbr_to_dataset,
    apply_mbr_to_dataset_asymmetric,
    apply_mbr_to_dataset_multi,
    mbr_with_weights,
)
from prompts.mbr_prompts import create_messages
from prompts.postprocess import postprocess_summary

DEFAULT_REGISTRY = LLM_DIR / "conf" / "checkpoints.yaml"

MAX_SEQ_LENGTH = 2048
MAX_NEW_TOKENS = 128


def _postprocess(text: str) -> str:
    return postprocess_summary(text)


def _resolve_path(p: str, base: Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (base / pp).resolve()


def load_checkpoint_registry(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("checkpoints", {})


def load_ensemble_config(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _adapter_has_saved_tokenizer_files(adapter_path: Path) -> bool:
    """학습 시 tokenizer.save_pretrained(adapter_dir)로 남긴 파일이 있는지(대략) 확인."""
    if not adapter_path.is_dir():
        return False
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "vocab.json",
        "special_tokens_map.json",
    )
    return any((adapter_path / n).is_file() for n in names)


def load_tokenizer_from_adapter(adapter_path: Path) -> Any:
    """
    체크포인트별로 학습에 쓰인 토크나이저만 사용합니다.

    베이스 모델 ID로 토크나이저를 불러오지 않고, 어댑터(또는 학습 산출물) 디렉터리에
    save_pretrained된 파일만 사용합니다. (run_qlora_*, inference_worker, run_prompts와 동일 정책)
    """
    adapter_path = adapter_path.resolve()
    if not _adapter_has_saved_tokenizer_files(adapter_path):
        raise FileNotFoundError(
            "어댑터 경로에 학습 시 저장된 토크나이저 파일이 없습니다. "
            "학습 스크립트에서 tokenizer.save_pretrained(lora_path)가 실행된 폴더인지 확인하세요.\n"
            f"  path={adapter_path}"
        )
    tok = AutoTokenizer.from_pretrained(
        str(adapter_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"  tokenizer ← {adapter_path} (학습 시 save_pretrained와 동일 경로, 베이스 ID로 로드하지 않음)")
    return tok


def load_lora_model(adapter_path: Path, base_model: str) -> Tuple[Any, Any]:
    """Unsloth 베이스 + LoRA. 토크나이저는 항상 어댑터 폴더(학습 시 저장본)에서만 로드."""
    from unsloth import FastLanguageModel

    adapter_path = adapter_path.resolve()
    base, _ = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    tok = load_tokenizer_from_adapter(adapter_path)
    model.eval()
    return model, tok


def _infer_base_from_registry(_adapter_path: Path, reg_entry: Dict[str, Any]) -> str:
    """레지스트리에 명시된 베이스 모델 id 반환."""
    return reg_entry.get("base_model", "unsloth/qwen3-14b-unsloth-bnb-4bit")


def generate_for_df(
    model,
    tokenizer,
    df: pd.DataFrame,
    prompt_name: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> List[str]:
    preds: List[str] = []
    device = next(model.parameters()).device
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"gen[{prompt_name}]"):
        topic = row["topic"] if "topic" in row and pd.notna(row["topic"]) else ""
        messages = create_messages(prompt_name, str(row["dialogue"]), topic=str(topic))
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        ).to(device)
        gen_kw: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
            "use_cache": False,  # Unsloth fast_forward_inference + PEFT 조합에서 KV cache shape 충돌
        }
        if do_sample:
            gen_kw["do_sample"] = True
            gen_kw["temperature"] = temperature
            gen_kw["top_p"] = top_p
        else:
            gen_kw["do_sample"] = False
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kw)
        gen = out[0][inputs["input_ids"].shape[1] :]
        raw = tokenizer.decode(gen, skip_special_tokens=True)
        preds.append(_postprocess(raw))
    return preds


def _combo_cache_path(out_dir: Path, combo_key: str) -> Path:
    safe = re.sub(r"[^\w\-.]+", "_", combo_key)
    return out_dir / "cache" / f"{safe}.csv"


def load_cache_csv(path: Path, n_expected: int) -> Optional[List[str]]:
    if not path.is_file():
        return None
    try:
        c = pd.read_csv(path)
        if len(c) != n_expected:
            return None
        if "summary" in c.columns:
            return c["summary"].tolist()
        if "pred_summary" in c.columns:
            return c["pred_summary"].tolist()
    except Exception:
        return None
    return None


def save_cache_csv(path: Path, df: pd.DataFrame, preds: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy() if "fname" in df.columns else pd.DataFrame({"idx": range(len(preds))})
    out["summary"] = preds
    out.to_csv(path, index=False)


def build_weights_keyed(
    ckpt_configs: List[Dict[str, Any]],
    prompt_names: List[str],
    mode: int,
) -> Dict[str, float]:
    w: Dict[str, float] = {}
    for c in ckpt_configs:
        name = c["name"]
        wt = float(c.get("weight", 1.0))
        if mode == 1:
            p = c.get("prompt", "qa_style")
            w[f"{name}__{p}"] = wt
        else:
            for p in prompt_names:
                w[f"{name}__{p}"] = wt
    return w


def apply_mbr_pipeline(
    test_df: pd.DataFrame,
    all_predictions: Dict[str, List[str]],
    mbr_cfg: Dict[str, Any],
    weights: Dict[str, float],
    no_weight: bool,
) -> List[str]:
    """utility / asymmetric / weighted 조합에 따라 MBR 적용."""
    utility = mbr_cfg.get("utility", "rouge_multi")
    if utility in ("bertscore", "combined"):
        utility = "bertscore_rouge_combined"
    use_mecab = mbr_cfg.get("use_mecab", True)
    asymmetric = mbr_cfg.get("asymmetric", False)
    ref_policy = mbr_cfg.get("reference_policy", "greedy")
    weighted = mbr_cfg.get("weighted", True) and not no_weight

    keys = list(all_predictions.keys())
    n = len(test_df)

    ref_keys: Optional[Set[str]] = None
    if asymmetric:
        if ref_policy == "all":
            ref_keys = None
        else:
            ref_keys = {k for k in keys if k.endswith("__g") or k.endswith("__greedy")}
            if not ref_keys:
                ref_keys = set(keys)

    w = weights if weighted else None

    if asymmetric and ref_keys is not None and ref_keys != set(keys):
        multi = ["rouge-1", "rouge-2", "rouge-l"] if utility in (
            "rouge_multi",
            "bertscore_rouge_combined",
        ) else None
        return apply_mbr_to_dataset_asymmetric(
            test_df,
            all_predictions,
            ref_keys,
            use_mecab=use_mecab,
            metric="rouge-1",
            multi_metrics=multi,
            weights=w,
            verbose=True,
        )

    if weighted and w and utility == "rouge1":
        mbr_preds = []
        model_selected = {name: 0 for name in keys}
        for i in tqdm(range(n), desc="MBR (weighted ROUGE-1)"):
            candidates = [(k, all_predictions[k][i]) for k in keys]
            selected = mbr_with_weights(candidates, weights=w, use_mecab=use_mecab, metric="rouge-1")
            mbr_preds.append(selected)
            for name, text in candidates:
                if text == selected:
                    model_selected[name] += 1
                    break
        print("\n" + "=" * 80)
        print("MBR 앙상블 결과 - 모델 선택 빈도 (weighted ROUGE-1)")
        print("=" * 80)
        for name, count in sorted(model_selected.items(), key=lambda x: -x[1]):
            pct = 100 * count / n
            print(f"  {name:40s}: {count:4d} ({pct:5.1f}%)")
        print("=" * 80)
        return mbr_preds

    if utility in ("rouge_multi", "bertscore_rouge_combined"):
        return apply_mbr_to_dataset_multi(
            test_df, all_predictions, use_mecab=use_mecab, verbose=True, weights=w if weighted else None
        )
    return apply_mbr_to_dataset(
        test_df,
        all_predictions,
        use_mecab=use_mecab,
        metric="rouge-1",
        verbose=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="멀티 체크포인트 MBR 앙상블")
    parser.add_argument("--config", type=str, required=True, help="ensemble_mode*.yaml")
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument("--prompts", nargs="*", default=None)
    parser.add_argument(
        "--utility",
        type=str,
        default=None,
        choices=["rouge1", "rouge_multi", "bertscore", "bertscore_rouge_combined", "combined"],
    )
    parser.add_argument("--asymmetric", action="store_true")
    parser.add_argument("--reference_policy", type=str, default=None)
    parser.add_argument("--no_mecab", action="store_true")
    parser.add_argument("--no_weight", action="store_true")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--save_all", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--checkpoint_registry", type=str, default=str(DEFAULT_REGISTRY))
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    cfg_path = _resolve_path(args.config, LLM_DIR)
    if not cfg_path.is_file():
        print(f"[오류] 설정 파일 없음: {cfg_path}")
        sys.exit(1)

    cfg = load_ensemble_config(cfg_path)
    reg_path = Path(args.checkpoint_registry)
    if not reg_path.is_absolute():
        reg_path = (LLM_DIR / reg_path).resolve()
    registry = load_checkpoint_registry(reg_path)

    mode = int(cfg.get("mode", 2))
    out_cfg = cfg.get("output", {})
    output_dir = LLM_DIR / out_cfg.get("output_dir", "outputs/ensemble_default/").lstrip("/")
    save_per_combo = args.save_all or out_cfg.get("save_per_combo", False)

    df = pd.read_csv(_resolve_path(args.test_file, Path.cwd()))
    if "dialogue" not in df.columns:
        print("[오류] test_file에 dialogue 컬럼이 필요합니다.")
        sys.exit(1)

    # 체크포인트 목록
    ckpt_cfgs = cfg.get("checkpoints", [])
    if args.checkpoints:
        names = set(args.checkpoints)
        ckpt_cfgs = [c for c in ckpt_cfgs if c["name"] in names]
    if not ckpt_cfgs:
        print("[오류] 사용할 체크포인트가 없습니다.")
        sys.exit(1)

    # 프롬프트
    if mode == 1:
        prompt_lists = [c.get("prompt", "qa_style") for c in ckpt_cfgs]
    else:
        plist = cfg.get("prompts", ["base", "qa_style"])
        if args.prompts:
            plist = list(args.prompts)
        prompt_lists = plist

    mbr_cfg = cfg.get("mbr", {})
    if args.utility:
        umap = {"combined": "bertscore_rouge_combined"}
        mbr_cfg["utility"] = umap.get(args.utility, args.utility)
    if args.asymmetric:
        mbr_cfg["asymmetric"] = True
    if args.reference_policy:
        mbr_cfg["reference_policy"] = args.reference_policy
    if args.no_mecab:
        mbr_cfg["use_mecab"] = False

    sampling = cfg.get("sampling", {})
    num_samples = args.num_samples if args.num_samples is not None else sampling.get("num_samples", 0)
    temperature = args.temperature if args.temperature is not None else sampling.get("temperature", 0.7)
    top_p = args.top_p if args.top_p is not None else sampling.get("top_p", 0.9)
    include_greedy = sampling.get("include_greedy", True)
    seed = sampling.get("seed", 42)

    if args.dry_run:
        print("[dry_run] 설정 요약")
        print(f"  mode={mode}  registry={reg_path}")
        for c in ckpt_cfgs:
            name = c["name"]
            ent = registry.get(name)
            if not ent:
                print(f"  [경고] 레지스트리에 없음: {name}")
                continue
            ap = _resolve_path(ent["path"], LLM_DIR)
            print(f"  - {name}: path={ap} exists={ap.is_dir()}")
        print(f"  prompts: {prompt_lists}")
        print(f"  MBR: {mbr_cfg}")
        print(f"  output_dir: {output_dir}")
        return

    if mbr_cfg.get("utility") == "bertscore_rouge_combined":
        print(
            "[안내] bertscore_rouge_combined: 현재 구현은 ROUGE-multi와 동일하게 동작합니다. "
            "(BERTScore 패키지 연동은 추후 확장)"
        )

    torch.manual_seed(seed)

    if mode == 3 and num_samples == 0:
        print("[경고] mode=3이지만 num_samples=0입니다. greedy-only로 동작합니다.")
    if mode == 3 and not include_greedy and mbr_cfg.get("asymmetric", False):
        print(
            "[경고] include_greedy=false + asymmetric=true: "
            "greedy 키(__g)가 없어 비대칭 MBR이 표준 MBR로 폴백됩니다."
        )

    all_predictions: Dict[str, List[str]] = {}
    n = len(df)

    for c in ckpt_cfgs:
        name = c["name"]
        ent = registry.get(name)
        if not ent:
            print(f"[경고] 건너뜀 — 레지스트리에 없음: {name}")
            continue
        adapter_path = _resolve_path(ent["path"], LLM_DIR)
        if not adapter_path.is_dir():
            print(f"[경고] 건너뜀 — 경로 없음: {adapter_path}")
            continue
        base_model = _infer_base_from_registry(adapter_path, ent)

        print(f"\n{'='*60}\n체크포인트 로드: {name}\n  {adapter_path}\n  base: {base_model}\n{'='*60}")

        model, tokenizer = load_lora_model(adapter_path, base_model)

        if mode == 1:
            prompts_here = [c.get("prompt", "qa_style")]
        else:
            prompts_here = prompt_lists if isinstance(prompt_lists, list) else list(prompt_lists)

        for pn in prompts_here:
            if mode == 3 and num_samples and num_samples > 0:
                if include_greedy:
                    gkey = f"{name}__{pn}__g"
                    cache_p = _combo_cache_path(output_dir, gkey)
                    cached = load_cache_csv(cache_p, n) if args.resume else None
                    if cached is not None:
                        all_predictions[gkey] = cached
                        print(f"  [resume] {gkey} 캐시 사용")
                    else:
                        preds = generate_for_df(
                            model,
                            tokenizer,
                            df,
                            pn,
                            args.max_new_tokens,
                            do_sample=False,
                            temperature=1.0,
                            top_p=1.0,
                        )
                        all_predictions[gkey] = preds
                        if save_per_combo:
                            save_cache_csv(cache_p, df, preds)

                for si in range(num_samples):
                    skey = f"{name}__{pn}__s{si}"
                    torch.manual_seed(seed + si)
                    cache_p = _combo_cache_path(output_dir, skey)
                    cached = load_cache_csv(cache_p, n) if args.resume else None
                    if cached is not None:
                        all_predictions[skey] = cached
                        print(f"  [resume] {skey} 캐시 사용")
                    else:
                        spreds = generate_for_df(
                            model,
                            tokenizer,
                            df,
                            pn,
                            args.max_new_tokens,
                            do_sample=True,
                            temperature=temperature,
                            top_p=top_p,
                        )
                        all_predictions[skey] = spreds
                        if save_per_combo:
                            save_cache_csv(cache_p, df, spreds)
            else:
                key = f"{name}__{pn}"
                cache_p = _combo_cache_path(output_dir, key)
                cached = load_cache_csv(cache_p, n) if args.resume else None
                if cached is not None:
                    all_predictions[key] = cached
                    print(f"  [resume] {key} 캐시 사용")
                else:
                    preds = generate_for_df(
                        model,
                        tokenizer,
                        df,
                        pn,
                        args.max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                    )
                    all_predictions[key] = preds
                    if save_per_combo:
                        save_cache_csv(cache_p, df, preds)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    if not all_predictions:
        print("[오류] 생성된 예측이 없습니다.")
        sys.exit(1)

    if mode == 1:
        weights = {
            f"{c['name']}__{c.get('prompt', 'qa_style')}": float(c.get("weight", 1.0))
            for c in ckpt_cfgs
        }
    else:
        weights = build_weights_keyed(ckpt_cfgs, prompt_lists, mode)

    final = apply_mbr_pipeline(df, all_predictions, mbr_cfg, weights, args.no_weight)

    out_path = _resolve_path(args.output_file, Path.cwd())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if "fname" in df.columns:
        pd.DataFrame({"fname": df["fname"], "summary": final}).to_csv(out_path, index=False)
    else:
        pd.DataFrame({"summary": final}).to_csv(out_path, index=False)

    if "summary" in df.columns:
        try:
            from rouge import Rouge

            rouge = Rouge()
            from prompts.mecab_ko import get_mecab

            m = get_mecab()
            preds_t = [" ".join(m.morphs(p)) if p.strip() else "빈요약" for p in final]
            golds_t = [" ".join(m.morphs(g)) if str(g).strip() else "빈요약" for g in df["summary"]]
            s = rouge.get_scores(preds_t, golds_t, avg=True)
            r1 = s["rouge-1"]["f"]
            print(f"\n[dev ROUGE-1] {r1:.4f}")
        except Exception as e:
            if args.verbose:
                print(f"[dev ROUGE 스킵] {e}")

    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
