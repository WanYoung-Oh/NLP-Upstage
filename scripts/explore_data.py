"""
데이터 탐색 (EDA) 스크립트.

train.csv / dev.csv / test.csv 각각에 대해 아래 항목을 분석하고
results/eda/ 디렉토리에 보고서(Markdown) 및 시각화(PNG)를 저장합니다.

분석 항목:
  1. 기본 통계 (행 수, 컬럼 구성, 결측치)
  2. dialogue 길이 분포 (문자 수 / 토큰 수 / 발화 수)
  3. summary 길이 분포 (train·dev 전용)
  4. topic 분포 (train·dev 전용)
  5. 노이즈 패턴 분석
       - 단독 자음/모음
       - 빈 괄호 ( (), [], {} )
       - 반복 특수기호 (!!!, ???, ~~~, ...)
  6. 스페셜 토큰 등장 빈도 (#Person1# ~ #Email# 등 15종)
  7. 발화자 수 분포 (대화당 등장 화자 수)

실행 예시:
    # 프로젝트 루트에서
    python scripts/explore_data.py

    # 데이터 경로 / 출력 경로 직접 지정
    python scripts/explore_data.py --data_dir data_aug --output_dir results/eda_aug

    # 시각화 저장 없이 터미널 출력만
    python scripts/explore_data.py --no_plot
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from datetime import datetime

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

SPECIAL_TOKENS: list[str] = [
    "#Person1#", "#Person2#", "#Person3#", "#Person4#",
    "#Person5#", "#Person6#", "#Person7#",
    "#PhoneNumber#", "#Address#", "#PassportNumber#",
    "#DateOfBirth#", "#SSN#", "#CardNumber#", "#CarNumber#", "#Email#",
]

# 노이즈 패턴 정규식
_PAT_CONSONANT = re.compile(r"[ㄱ-ㅎ]+")          # 단독 자음
_PAT_VOWEL     = re.compile(r"[ㅏ-ㅣ]+")           # 단독 모음
_PAT_EMPTY_BR  = re.compile(r"\(\s*\)|\[\s*\]|\{\s*\}")  # 빈 괄호
_PAT_REPEAT    = re.compile(r"([^\w\s#])\1{2,}")   # 반복 특수기호 (3회+)


# ---------------------------------------------------------------------------
# 노이즈 카운터
# ---------------------------------------------------------------------------

def _count_noise(text: str) -> dict[str, int]:
    """단일 텍스트에서 노이즈 패턴 등장 횟수를 반환합니다."""
    return {
        "consonant":  len(_PAT_CONSONANT.findall(text)),
        "vowel":      len(_PAT_VOWEL.findall(text)),
        "empty_br":   len(_PAT_EMPTY_BR.findall(text)),
        "repeat_sym": len(_PAT_REPEAT.findall(text)),
    }


def _count_special_tokens(text: str) -> dict[str, int]:
    """텍스트에서 스페셜 토큰별 등장 횟수를 반환합니다."""
    return {tok: text.count(tok) for tok in SPECIAL_TOKENS}


def _speaker_count(dialogue: str) -> int:
    """대화에서 등장하는 고유 화자 수를 반환합니다."""
    found = set(re.findall(r"(#Person\d+#)", dialogue))
    return len(found)


def _utterance_count(dialogue: str) -> int:
    """대화 발화 수(줄 수)를 반환합니다."""
    return len([l for l in dialogue.split("\n") if l.strip()])


# ---------------------------------------------------------------------------
# 데이터셋별 분석
# ---------------------------------------------------------------------------

def analyze(df: pd.DataFrame, name: str) -> dict:
    """DataFrame 하나를 분석하여 통계 dict를 반환합니다."""
    stats: dict = {"name": name, "n_rows": len(df), "columns": df.columns.tolist()}

    # ── 결측치 ────────────────────────────────────────────────────────────
    stats["null_counts"] = df.isnull().sum().to_dict()

    # ── dialogue 길이 ─────────────────────────────────────────────────────
    dial = df["dialogue"].fillna("")
    char_len = dial.str.len()
    utt_cnt  = dial.apply(_utterance_count)
    spk_cnt  = dial.apply(_speaker_count)

    stats["dialogue_char"] = char_len.describe().round(1).to_dict()
    stats["dialogue_utt"]  = utt_cnt.describe().round(1).to_dict()
    stats["dialogue_spk"]  = spk_cnt.value_counts().sort_index().to_dict()
    stats["dialogue_char_raw"] = char_len.tolist()
    stats["dialogue_utt_raw"]  = utt_cnt.tolist()

    # ── summary 길이 (train / dev 전용) ───────────────────────────────────
    if "summary" in df.columns:
        summ = df["summary"].fillna("")
        summ_len = summ.str.len()
        stats["summary_char"] = summ_len.describe().round(1).to_dict()
        stats["summary_char_raw"] = summ_len.tolist()
        # 극단값 탐지
        stats["summary_too_short"] = int((summ_len < 10).sum())
        stats["summary_too_long"]  = int((summ_len > 250).sum())

    # ── topic 분포 (train / dev 전용) ─────────────────────────────────────
    if "topic" in df.columns:
        topic_vc = df["topic"].fillna("(결측)").value_counts()
        stats["topic_unique"] = int(topic_vc.shape[0])
        stats["topic_once"]   = int((topic_vc == 1).sum())
        stats["topic_top30"]  = topic_vc.head(30).to_dict()
        stats["topic_freq_dist"] = topic_vc.value_counts().sort_index().head(20).to_dict()
        stats["topic_raw"]    = df["topic"].fillna("(결측)").tolist()

    # ── 노이즈 패턴 분석 ──────────────────────────────────────────────────
    noise_rows = dial.apply(_count_noise)
    noise_df = pd.DataFrame(noise_rows.tolist())

    stats["noise"] = {}
    for col in ["consonant", "vowel", "empty_br", "repeat_sym"]:
        affected = int((noise_df[col] > 0).sum())
        total_hit = int(noise_df[col].sum())
        stats["noise"][col] = {
            "affected_rows": affected,
            "affected_pct":  round(affected / len(df) * 100, 2),
            "total_hits":    total_hit,
        }

    # ── 스페셜 토큰 분석 ──────────────────────────────────────────────────
    tok_rows = dial.apply(_count_special_tokens)
    tok_df = pd.DataFrame(tok_rows.tolist())

    stats["special_tokens"] = {}
    for tok in SPECIAL_TOKENS:
        if tok in tok_df.columns:
            cnt = int(tok_df[tok].sum())
            rows = int((tok_df[tok] > 0).sum())
            stats["special_tokens"][tok] = {"total": cnt, "rows_with": rows}

    return stats


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------

def plot_all(all_stats: list[dict], output_dir: str) -> None:
    """전체 분석 결과를 시각화하여 PNG로 저장합니다."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        import numpy as np
    except ImportError:
        print("[경고] matplotlib 미설치 — 시각화를 건너뜁니다. pip install matplotlib")
        return

    # 한국어 폰트 설정 (없으면 기본 폰트 사용)
    _setup_korean_font()

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Dialogue char length distribution ─────────────────────────────
    fig, axes = plt.subplots(1, len(all_stats), figsize=(6 * len(all_stats), 4), squeeze=False)
    for ax, s in zip(axes[0], all_stats):
        data = s.get("dialogue_char_raw", [])
        ax.hist(data, bins=50, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.set_title(f"{s['name']} — Dialogue Char Length", fontsize=11)
        ax.set_xlabel("Char Count")
        ax.set_ylabel("Frequency")
        mean_v = sum(data) / len(data) if data else 0
        ax.axvline(mean_v, color="tomato", linestyle="--", label=f"mean {mean_v:.0f}")
        ax.axvline(1500, color="orange", linestyle=":", label="filter threshold 1500")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dialogue_char_dist.png"), dpi=150)
    plt.close()

    # ── 2. Utterance count distribution ──────────────────────────────────
    fig, axes = plt.subplots(1, len(all_stats), figsize=(6 * len(all_stats), 4), squeeze=False)
    for ax, s in zip(axes[0], all_stats):
        data = s.get("dialogue_utt_raw", [])
        ax.hist(data, bins=30, color="mediumseagreen", edgecolor="white", linewidth=0.3)
        ax.set_title(f"{s['name']} — Utterance Count", fontsize=11)
        ax.set_xlabel("Utterances per Dialogue")
        ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "utterance_count_dist.png"), dpi=150)
    plt.close()

    # ── 3. Summary char length distribution (train / dev) ────────────────
    has_summary = [s for s in all_stats if "summary_char_raw" in s]
    if has_summary:
        fig, axes = plt.subplots(1, len(has_summary), figsize=(6 * len(has_summary), 4), squeeze=False)
        for ax, s in zip(axes[0], has_summary):
            data = s["summary_char_raw"]
            ax.hist(data, bins=40, color="mediumpurple", edgecolor="white", linewidth=0.3)
            ax.set_title(f"{s['name']} — Summary Char Length", fontsize=11)
            ax.set_xlabel("Char Count")
            ax.set_ylabel("Frequency")
            mean_v = sum(data) / len(data) if data else 0
            ax.axvline(mean_v, color="tomato", linestyle="--", label=f"mean {mean_v:.0f}")
            ax.axvline(10, color="orange", linestyle=":", label="min threshold 10")
            ax.axvline(250, color="orange", linestyle=":", label="max threshold 250")
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "summary_char_dist.png"), dpi=150)
        plt.close()

    # ── 4. Topic top-30 bar chart (train / dev) ───────────────────────────
    has_topic = [s for s in all_stats if "topic_top30" in s]
    for s in has_topic:
        top30 = s["topic_top30"]
        labels = list(top30.keys())
        values = list(top30.values())
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(labels[::-1], values[::-1], color="cornflowerblue")
        ax.set_title(f"{s['name']} — Top 30 Topics", fontsize=12)
        ax.set_xlabel("Count")
        for bar, val in zip(bars, values[::-1]):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"topic_top30_{s['name']}.png"), dpi=150)
        plt.close()

    # ── 5. Noise pattern comparison bar chart ────────────────────────────
    noise_labels = {
        "consonant":  "Lone Consonant",
        "vowel":      "Lone Vowel",
        "empty_br":   "Empty Bracket",
        "repeat_sym": "Repeated Symbol",
    }
    x = range(len(noise_labels))
    width = 0.8 / max(len(all_stats), 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, s in enumerate(all_stats):
        pcts = [s["noise"].get(k, {}).get("affected_pct", 0) for k in noise_labels]
        offset = (i - len(all_stats) / 2 + 0.5) * width
        bars = ax.bar([xi + offset for xi in x], pcts, width=width * 0.9, label=s["name"])
        for bar, pct in zip(bars, pcts):
            if pct > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(noise_labels.values()))
    ax.set_ylabel("Affected Rows (%)")
    ax.set_title("Noise Pattern — Affected Row Ratio")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "noise_pattern.png"), dpi=150)
    plt.close()

    # ── 6. Special token frequency bar chart ─────────────────────────────
    fig, axes = plt.subplots(1, len(all_stats), figsize=(7 * len(all_stats), 5), squeeze=False)
    for ax, s in zip(axes[0], all_stats):
        tok_data = s.get("special_tokens", {})
        tokens = list(tok_data.keys())
        totals = [tok_data[t]["total"] for t in tokens]
        colors = ["steelblue" if t.startswith("#Person") else "darkorange" for t in tokens]
        ax.bar(range(len(tokens)), totals, color=colors)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels([t.replace("#", "") for t in tokens], rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{s['name']} — Special Token Frequency", fontsize=11)
        ax.set_ylabel("Total Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "special_token_freq.png"), dpi=150)
    plt.close()

    # ── 7. Speaker count distribution ────────────────────────────────────
    fig, axes = plt.subplots(1, len(all_stats), figsize=(5 * len(all_stats), 4), squeeze=False)
    for ax, s in zip(axes[0], all_stats):
        spk = s.get("dialogue_spk", {})
        if spk:
            xs = sorted(spk.keys())
            ys = [spk[k] for k in xs]
            ax.bar(xs, ys, color="teal")
            for xi, yi in zip(xs, ys):
                ax.text(xi, yi + max(ys) * 0.01, str(yi), ha="center", fontsize=8)
        ax.set_title(f"{s['name']} — Speakers per Dialogue", fontsize=11)
        ax.set_xlabel("Speaker Count")
        ax.set_ylabel("Dialogue Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "speaker_count_dist.png"), dpi=150)
    plt.close()

    print(f"[Plot] 7 PNG files saved -> {output_dir}/")


def _setup_korean_font() -> None:
    """시스템에 한국어 폰트가 있으면 matplotlib에 등록합니다."""
    try:
        import matplotlib.font_manager as fm
        import matplotlib.pyplot as plt

        # 후보 폰트 목록 (Linux/Mac/Windows 순)
        candidates = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "C:/Windows/Fonts/malgun.ttf",
        ]
        for path in candidates:
            if os.path.exists(path):
                fm.fontManager.addfont(path)
                prop = fm.FontProperties(fname=path)
                plt.rcParams["font.family"] = prop.get_name()
                plt.rcParams["axes.unicode_minus"] = False
                return
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Markdown 보고서 생성
# ---------------------------------------------------------------------------

def _fmt_desc(d: dict) -> str:
    """describe() dict를 읽기 좋은 문자열로 변환합니다."""
    keys = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    parts = []
    for k in keys:
        if k in d:
            parts.append(f"{k}={d[k]:.1f}" if isinstance(d[k], float) else f"{k}={d[k]}")
    return "  |  ".join(parts)


def write_report(all_stats: list[dict], output_dir: str, has_plot: bool) -> None:
    """Markdown 형식의 EDA 보고서를 생성합니다."""
    os.makedirs(output_dir, exist_ok=True)
    lines: list[str] = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines += [
        f"# 데이터 EDA 보고서",
        f"",
        f"> 생성 일시: {ts}",
        f"",
        "---",
        "",
    ]

    for s in all_stats:
        name = s["name"]
        n    = s["n_rows"]
        cols = s["columns"]

        lines += [f"## {name}", ""]

        # ── 기본 통계 ────────────────────────────────────────────────────
        lines += ["### 기본 통계", ""]
        lines += [f"- **행 수**: {n:,}"]
        lines += [f"- **컬럼**: {', '.join(cols)}"]
        null = s["null_counts"]
        null_str = ", ".join(f"`{c}`: {v}" for c, v in null.items() if v > 0)
        lines += [f"- **결측치**: {null_str if null_str else '없음'}"]
        lines += [""]

        # ── dialogue 길이 ─────────────────────────────────────────────────
        lines += ["### Dialogue 길이 분포", ""]
        lines += [f"**문자 수**: {_fmt_desc(s['dialogue_char'])}"]
        lines += [f"**발화 수**: {_fmt_desc(s['dialogue_utt'])}"]
        lines += [""]
        lines += ["**화자 수 분포**:"]
        lines += ["| 화자 수 | 대화 수 | 비율 |"]
        lines += ["|---------|---------|------|"]
        spk = s.get("dialogue_spk", {})
        for k in sorted(spk.keys()):
            pct = spk[k] / n * 100
            lines += [f"| {k} | {spk[k]:,} | {pct:.1f}% |"]
        lines += [""]
        if has_plot:
            lines += [f"![dialogue 문자 길이](dialogue_char_dist.png)"]
            lines += [f"![발화 수 분포](utterance_count_dist.png)"]
            lines += [f"![화자 수 분포](speaker_count_dist.png)"]
            lines += [""]

        # ── summary 길이 ──────────────────────────────────────────────────
        if "summary_char" in s:
            lines += ["### Summary 길이 분포", ""]
            lines += [f"**문자 수**: {_fmt_desc(s['summary_char'])}"]
            lines += [f"- 10자 미만 (너무 짧음): **{s['summary_too_short']:,}건**"]
            lines += [f"- 250자 초과 (너무 김): **{s['summary_too_long']:,}건**"]
            lines += [""]
            if has_plot:
                lines += [f"![summary 문자 길이](summary_char_dist.png)", ""]

        # ── topic 분포 ────────────────────────────────────────────────────
        if "topic_unique" in s:
            lines += ["### Topic 분포", ""]
            lines += [f"- **고유 topic 수**: {s['topic_unique']:,}개 / {n:,}건 ({s['topic_unique']/n*100:.1f}%)"]
            lines += [f"- **1회만 등장**: {s['topic_once']:,}개 ({s['topic_once']/s['topic_unique']*100:.1f}%)"]
            lines += [f"- **2회 이상 등장**: {s['topic_unique'] - s['topic_once']:,}개"]
            lines += [""]
            lines += ["**상위 30개 topic**:"]
            lines += ["| 순위 | topic | 등장 수 |"]
            lines += ["|------|-------|---------|"]
            for rank, (tp, cnt) in enumerate(s["topic_top30"].items(), 1):
                lines += [f"| {rank} | {tp} | {cnt:,} |"]
            lines += [""]

            # topic 빈도 분포
            freq_dist = s.get("topic_freq_dist", {})
            if freq_dist:
                lines += ["**등장 횟수별 topic 수**:"]
                lines += ["| 등장 횟수 | topic 수 |"]
                lines += ["|-----------|---------|"]
                for freq, cnt in sorted(freq_dist.items()):
                    lines += [f"| {freq}회 | {cnt:,}개 |"]
                lines += [""]

            if has_plot:
                lines += [f"![topic 상위 30개](topic_top30_{name}.png)", ""]

        # ── 노이즈 패턴 ───────────────────────────────────────────────────
        lines += ["### 노이즈 패턴 분석 (dialogue 기준)", ""]
        lines += ["| 패턴 | 영향받은 행 | 비율 | 총 등장 횟수 |"]
        lines += ["|------|------------|------|-------------|"]
        noise_labels = {
            "consonant":  "단독 자음 (ㄱ-ㅎ)",
            "vowel":      "단독 모음 (ㅏ-ㅣ)",
            "empty_br":   "빈 괄호 ( (), [], {} )",
            "repeat_sym": "반복 특수기호 (3회+)",
        }
        for key, label in noise_labels.items():
            nd = s["noise"].get(key, {})
            lines += [
                f"| {label} | {nd.get('affected_rows', 0):,}행 "
                f"| {nd.get('affected_pct', 0):.2f}% "
                f"| {nd.get('total_hits', 0):,}회 |"
            ]
        lines += [""]
        if has_plot:
            lines += [f"![노이즈 패턴](noise_pattern.png)", ""]

        # ── 스페셜 토큰 ───────────────────────────────────────────────────
        lines += ["### 스페셜 토큰 등장 빈도 (dialogue 기준)", ""]
        lines += ["| 토큰 | 총 등장 횟수 | 등장 대화 수 | 비율 |"]
        lines += ["|------|------------|-------------|------|"]
        tok_data = s.get("special_tokens", {})
        for tok in SPECIAL_TOKENS:
            td = tok_data.get(tok, {"total": 0, "rows_with": 0})
            pct = td["rows_with"] / n * 100
            lines += [f"| `{tok}` | {td['total']:,} | {td['rows_with']:,} | {pct:.1f}% |"]
        lines += [""]
        if has_plot:
            lines += [f"![스페셜 토큰 빈도](special_token_freq.png)", ""]

        lines += ["---", ""]

    report_path = os.path.join(output_dir, "eda_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[보고서] → {report_path}")


# ---------------------------------------------------------------------------
# 터미널 요약 출력
# ---------------------------------------------------------------------------

def print_summary(all_stats: list[dict]) -> None:
    """핵심 통계를 터미널에 요약 출력합니다."""
    SEP = "=" * 70

    for s in all_stats:
        name = s["name"]
        n    = s["n_rows"]
        print(f"\n{SEP}")
        print(f"  {name}  ({n:,}행, 컬럼: {', '.join(s['columns'])})")
        print(SEP)

        # dialogue 길이
        dc = s["dialogue_char"]
        print(f"\n[Dialogue 문자 길이]  평균={dc.get('mean',0):.0f}  "
              f"중앙값={dc.get('50%',0):.0f}  "
              f"max={dc.get('max',0):.0f}  "
              f"1500자 초과={sum(1 for x in s['dialogue_char_raw'] if x > 1500):,}건")

        du = s["dialogue_utt"]
        print(f"[Dialogue 발화 수]    평균={du.get('mean',0):.1f}  "
              f"중앙값={du.get('50%',0):.0f}  "
              f"max={du.get('max',0):.0f}")

        # 화자 수
        spk = s.get("dialogue_spk", {})
        spk_str = "  ".join(f"{k}명:{v:,}건" for k, v in sorted(spk.items()))
        print(f"[화자 수 분포]         {spk_str}")

        # summary
        if "summary_char" in s:
            sc = s["summary_char"]
            print(f"\n[Summary 문자 길이]   평균={sc.get('mean',0):.0f}  "
                  f"중앙값={sc.get('50%',0):.0f}  "
                  f"max={sc.get('max',0):.0f}  "
                  f"<10자={s['summary_too_short']:,}건  >250자={s['summary_too_long']:,}건")

        # topic
        if "topic_unique" in s:
            print(f"\n[Topic]  고유={s['topic_unique']:,}  1회 등장={s['topic_once']:,}  "
                  f"상위3: {list(s['topic_top30'].items())[:3]}")

        # 노이즈
        print("\n[노이즈 패턴]")
        noise_labels = {
            "consonant":  "단독 자음",
            "vowel":      "단독 모음",
            "empty_br":   "빈 괄호",
            "repeat_sym": "반복 특수기호(3회+)",
        }
        for k, label in noise_labels.items():
            nd = s["noise"].get(k, {})
            print(f"  {label:<20s}  영향 {nd.get('affected_rows',0):>5,}행 "
                  f"({nd.get('affected_pct',0):.2f}%)  "
                  f"총 {nd.get('total_hits',0):,}회")

        # 스페셜 토큰 (총계 > 0인 것만)
        tok_data = s.get("special_tokens", {})
        found = {t: d for t, d in tok_data.items() if d["total"] > 0}
        if found:
            print("\n[스페셜 토큰] (등장한 것만)")
            for tok, d in found.items():
                print(f"  {tok:<20s}  총 {d['total']:>6,}회  "
                      f"{d['rows_with']:,}개 대화 ({d['rows_with']/n*100:.1f}%)")

    print(f"\n{SEP}\n")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="train/dev/test CSV 데이터 탐색 스크립트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir",   default="data",        help="CSV 파일 디렉토리")
    parser.add_argument("--output_dir", default="results/eda", help="보고서 및 시각화 저장 디렉토리")
    parser.add_argument("--no_plot",    action="store_true",    help="시각화 저장 생략 (터미널 출력만)")
    args = parser.parse_args()

    data_dir   = os.path.join(_ROOT, args.data_dir)
    output_dir = os.path.join(_ROOT, args.output_dir)

    # ── 데이터 로드 ──────────────────────────────────────────────────────
    csv_files = [
        ("train", os.path.join(data_dir, "train.csv")),
        ("dev",   os.path.join(data_dir, "dev.csv")),
        ("test",  os.path.join(data_dir, "test.csv")),
    ]

    all_stats: list[dict] = []
    for name, path in csv_files:
        if not os.path.exists(path):
            print(f"[건너뜀] {path} 없음")
            continue
        print(f"[로드] {path}")
        df = pd.read_csv(path)
        stats = analyze(df, name)
        all_stats.append(stats)

    if not all_stats:
        print("[오류] 분석할 CSV 파일이 없습니다.")
        sys.exit(1)

    # ── 터미널 요약 출력 ──────────────────────────────────────────────────
    print_summary(all_stats)

    # ── Markdown 보고서 저장 ──────────────────────────────────────────────
    write_report(all_stats, output_dir, has_plot=not args.no_plot)

    # ── 시각화 저장 ───────────────────────────────────────────────────────
    if not args.no_plot:
        plot_all(all_stats, output_dir)

    print(f"\n완료. 결과 → {output_dir}/")


if __name__ == "__main__":
    main()
