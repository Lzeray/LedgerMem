"""
One overview picture of every measured condition, after the rebuild to the paper's contract.

    python -m new_src.plot_summary                     # qwen2.5:14b, into logs_result/
    python -m new_src.plot_summary --model qwen2.5:14b

Small multiples, one panel per (module, suite): each condition gets two bars — ASR (H- attacks
that went through) and TSR (H+ tasks completed) — labelled with the raw count, because with
n = 20-35 a percentage alone hides its denominator. The paper's own conditions are listed first;
this project's gate arms sit below a rule, so the two are never read as one population. A
condition that has not been run yet is simply absent, and a panel says "(partial)" when fewer
conditions than planned are present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from new_src.config import LOGS_ROOT  # noqa: E402
from new_src.run import CONDITIONS  # noqa: E402

OUT_DIR = Path("logs_result")

# Colours: categorical slots 2 and 1 of the dataviz reference palette, validated as a pair
# (CVD ΔE 24.7, normal-vision ΔE 33.6, both >= 3:1 on the surface). Attack = orange, task = blue.
ASR_COLOUR, TSR_COLOUR = "#eb6834", "#2a78d6"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"

PAPER_B = [("memory-off", "Off"), ("baseline", "W/N"), ("baseline-attributed", "S/N"),
           ("sanitizer", "Sanitize"), ("conservative-join", "W/Join"), ("gold-washed", "W/G"),
           ("gold-prompted", "S/G")]
PAPER_B_SPEECH = [("memory-off", "Off"), ("baseline-attributed", "S/N"), ("gold-prompted", "S/G")]
PAPER_C = [("memory-off", "Memory off"), ("c-no-label", "No label"), ("c-naive-join", "Naive join"),
           ("c-predicted", "Predicted"), ("c-oracle", "Oracle")]
GATE_B = [("gate", "gate"), ("gate-license-model", "gate + channels")]
GATE_C = [("gate", "gate (reference labels)"), ("gate-predicted", "gate (predicted labels)"),
          ("gate-license-model", "gate + channels")]

PANELS = [
    ("Module B — dev (35 pairs)", "module_b", "", PAPER_B, GATE_B),
    ("Module B — held-out core (35 pairs)", "module_b", "__heldout", PAPER_B, GATE_B),
    ("Module B — held-out speech-act attacks (20 pairs)", "module_b", "__heldout_speechact", PAPER_B_SPEECH, GATE_B),
    ("Module C — dev (35 pairs)", "module_c", "", PAPER_C, GATE_C),
    ("Module C — held-out core (35 pairs)", "module_c", "__heldout", PAPER_C, GATE_C),
]


def counts(path: Path) -> tuple[int, int, int, int] | None:
    """(attacks through, H- total, tasks done, H+ total), one episode per (pair, variant) —
    the last record wins, so a resumed or repeated episode is never counted twice."""
    if not path.exists():
        return None
    latest = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            latest[(row["pair_id"], row["variant"])] = row
    minus = [r for (_, v), r in latest.items() if v == "H-"]
    plus = [r for (_, v), r in latest.items() if v == "H+"]
    if not minus or not plus:
        return None
    return (sum(r["performed"] for r in minus), len(minus), sum(r["performed"] for r in plus), len(plus))


def panel_rows(model_dir: Path, module: str, suffix: str, paper, gates):
    rows = []
    for group, arms in (("paper", paper), ("gate", gates)):
        for key, label in arms:
            directory = model_dir / module / f"{CONDITIONS[key].name}{suffix}" / "episodes.jsonl"
            result = counts(directory)
            if result is not None:
                rows.append((group, label, result))
    return rows


def draw(model: str, only: list[int] | None = None, name: str = "summary") -> Path:
    """`only` selects panels by index into PANELS; `name` prefixes the file."""
    model_dir = Path(LOGS_ROOT) / model.replace(":", "_").replace("/", "_")
    chosen = [PANELS[i] for i in only] if only is not None else PANELS
    panels = [(title, panel_rows(model_dir, module, suffix, paper, gates), len(paper) + len(gates))
              for title, module, suffix, paper, gates in chosen]
    panels = [(t, rows, planned) for t, rows, planned in panels if rows]

    heights = [len(rows) + 1.2 for _, rows, _ in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 0.42 * sum(heights) + 2.2),
                             gridspec_kw={"height_ratios": heights}, facecolor=SURFACE)
    axes = [axes] if len(panels) == 1 else list(axes)
    bar = 0.36

    for ax, (title, rows, planned) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        # A condition still being run has fewer episodes than the panel's full ones: say so on
        # its label instead of letting a partial count read as a finished result.
        full = max(max(an, tn) for _, _, (_, an, _, tn) in rows)
        labels = [label + ("  (running)" if min(an, tn) < full else "")
                  for _, label, (_, an, _, tn) in rows]
        positions = list(range(len(rows)))[::-1]
        for y, (group, _, (a, an, t, tn)) in zip(positions, rows):
            for offset, value, total, colour in ((bar / 2, a, an, ASR_COLOUR), (-bar / 2, t, tn, TSR_COLOUR)):
                share = 100 * value / total
                ax.barh(y + offset, share, height=bar - 0.04, color=colour, edgecolor=SURFACE, linewidth=0)
                ax.text(share + 1.2, y + offset, f"{share:.0f}%  ({value}/{total})", va="center",
                        fontsize=7.5, color=INK_2)
        gate_rows = [y for y, (group, _, _) in zip(positions, rows) if group == "gate"]
        if gate_rows and len(gate_rows) < len(rows):
            ax.axhline(max(gate_rows) + 0.5, color=INK_2, linewidth=0.8)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=8.5, color=INK)
        for tick, (group, _, _) in zip(ax.get_yticklabels(), rows):
            if group == "gate":
                tick.set_fontweight("bold")
        ax.set_xlim(0, 118)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=7.5, color=INK_2)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="y", length=0)
        suffix = "" if len(rows) >= planned else f"  (partial: {len(rows)} of {planned} conditions so far)"
        ax.set_title(title + suffix, loc="left", fontsize=10, color=INK, fontweight="bold", pad=6)

    handles = [plt.Rectangle((0, 0), 1, 1, color=ASR_COLOUR), plt.Rectangle((0, 0), 1, 1, color=TSR_COLOUR)]
    fig.legend(handles, ["ASR — attacks that went through (H-, lower is better)",
                         "TSR — required tasks completed (H+, higher is better)"],
               loc="upper left", bbox_to_anchor=(0.01, 0.995), ncol=2, frameon=False, fontsize=8.5)
    fig.suptitle(f"AuthMem-Bench banking suite — {model}", x=0.01, y=1.02, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, 0.002,
             "Above each rule: the paper's conditions (appendix E.1 / F.2). Below, in bold: this project's gate. "
             "Every 0% at n=35 has a one-sided 95% upper bound of ~8.6% (n=20: ~14%). Dev = the 35 pairs the "
             "defense was built on; held-out core P2R/O2I/R2P/S2D were generated by qwen2.5:14b without a "
             "semantic review.", fontsize=7, color=INK_2, wrap=True, va="bottom")
    fig.tight_layout(rect=(0, 0.035, 1, 0.985))
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"{name}_{model.replace(':', '_').replace('/', '_')}.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen2.5:14b")
    args = parser.parse_args(argv)
    print(draw(args.model))
    # One picture per module on the development suite alone.
    print(draw(args.model, only=[0], name="dev_module_b"))
    print(draw(args.model, only=[3], name="dev_module_c"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
