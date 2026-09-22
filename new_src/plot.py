"""
Draw the results table as an image.

    python -m new_src.plot                 # every module found, into logs_result/
    python -m new_src.plot --module c      # just Module C

One picture per module. Rows are the attack categories; each condition gets two columns —
how many H- attacks got through, and how many H+ tasks were completed — both as a count out
of the pairs actually run. The bottom row is the same totals as percentages: ASR on the H-
side, TSR on the H+ side.

Counts come first and percentages second on purpose. With five pairs per category a
percentage hides its own denominator, and a category whose pairs failed the null control, or
never ran, looks identical to one that scored zero honestly.

Pairs that failed the null control are marked, not silently dropped: their H- side proves
nothing about authority, so the ASR total is computed without them while the table still
shows they exist.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from new_src.bench.metrics import load_jsonl  # noqa: E402
from new_src.config import LOGS_ROOT  # noqa: E402

OUT_DIR = Path("logs_result")
CATEGORIES = ["R2F", "P2R", "C2O", "MIX", "O2I", "R2P", "S2D"]
#: Per module: (condition key in run.CONDITIONS, column name). The paper's conditions first,
#: this project's gate arms last. Memory-off is left out of the table (it is 0/0 everywhere by
#: construction) and stated in the footnote instead.
ARMS = {
    "b": [("baseline", "W/N"), ("baseline-attributed", "S/N"), ("sanitizer", "Sanitize"),
          ("conservative-join", "W/Join"), ("gold-washed", "W/G"), ("gold-prompted", "S/G"),
          ("gate", "gate"), ("gate-license-model", "gate+channels")],
    "c": [("c-no-label", "No label"), ("c-naive-join", "Naive join"), ("c-predicted", "Predicted"),
          ("c-oracle", "Oracle"), ("gate-license-model", "gate+channels")],
}
GATE_KEYS = {"gate", "gate-license-model"}


def _by_pair(rows: list[dict], variant: str) -> dict[str, float]:
    """One value per pair, averaging any repeated runs of the same episode."""
    grouped = defaultdict(list)
    for row in rows:
        if row["variant"] == variant:
            grouped[row["pair_id"]].append(bool(row["performed"]))
    return {pair: sum(v) / len(v) for pair, v in grouped.items()}


def collect(model_dir: Path, module: str, suffix: str) -> tuple[dict, set[str]]:
    """{column: {category: (h_minus_through, h_minus_total, h_plus_done, h_plus_total)}}, read
    from the exact directory each condition writes to, plus the pairs whose null control fired."""
    from new_src.run import CONDITIONS

    nulls: set[str] = set()
    null_dir = "null_control__heldout" if suffix else "null_control"
    for path in model_dir.rglob(f"{null_dir}*/episodes.jsonl"):
        if path.parent.name == null_dir or path.parent.name.startswith("baseline_without"):
            nulls |= {r["pair_id"] for r in load_jsonl(path) if r["performed"]}
    if not suffix:
        for path in (model_dir / "null_control").rglob("episodes.jsonl") if (model_dir / "null_control").exists() else []:
            nulls |= {r["pair_id"] for r in load_jsonl(path) if r["performed"]}

    out: dict[str, dict] = {}
    for key, label in ARMS[module]:
        path = model_dir / f"module_{module}" / f"{CONDITIONS[key].name}{suffix}" / "episodes.jsonl"
        if not path.exists():
            continue
        rows = load_jsonl(path)
        minus, plus = _by_pair(rows, "H-"), _by_pair(rows, "H+")
        per_category = {}
        for category in CATEGORIES:
            m = {k: v for k, v in minus.items() if k.endswith(category)}
            p = {k: v for k, v in plus.items() if k.endswith(category)}
            per_category[category] = (sum(m.values()), len(m), sum(p.values()), len(p))
        out[label] = per_category
    return out, nulls


def draw(data: dict, nulls: set[str], title: str, path: Path, module: str) -> None:
    arms = [label for _, label in ARMS[module] if label in data]
    gate_labels = {label for key, label in ARMS[module] if key in GATE_KEYS}
    fig, ax = plt.subplots(figsize=(3.0 + 2.6 * len(arms), 5.2))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=18)

    header = ["category"]
    for label in arms:
        header += [f"{label}\nH−  through", f"{label}\nH+  done"]
    body, colours = [], []
    for category in CATEGORIES:
        row = [category]
        tint = ["#f5f5f5"]
        for label in arms:
            through, m_total, done, p_total = data[label][category]
            row += [f"{through:g}/{m_total}" if m_total else "—",
                    f"{done:g}/{p_total}" if p_total else "—"]
            # Red when attacks got through, green when none did; green when tasks completed.
            tint.append("#f8d7da" if through else ("#d4edda" if m_total else "#ffffff"))
            tint.append("#d4edda" if p_total and done == p_total else
                        ("#fff3cd" if p_total else "#ffffff"))
        body.append(row)
        colours.append(tint)

    totals = ["ASR / TSR"]
    tint = ["#eeeeee"]
    for label in arms:
        m_through = sum(v[0] for k, v in data[label].items())
        m_total = sum(v[1] for k, v in data[label].items())
        # ASR excludes pairs whose null control fired: their H- side is not evidence about
        # authority. TSR keeps every pair, because the null control says nothing about H+.
        clean = {k: v for k, v in data[label].items()}
        p_done = sum(v[2] for v in clean.values())
        p_total = sum(v[3] for v in clean.values())
        totals += [f"{100 * m_through / m_total:.1f}%" if m_total else "—",
                   f"{100 * p_done / p_total:.1f}%" if p_total else "—"]
        tint += ["#eeeeee", "#eeeeee"]
    body.append(totals)
    colours.append(tint)

    table = ax.table(cellText=body, colLabels=header, cellColours=colours,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.9)
    for (row, column), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_height(cell.get_height() * 1.6)
            # This project's gate columns get a darker header, so they are never read as one of
            # the paper's conditions.
            arm = arms[(column - 1) // 2] if column else None
            cell.set_facecolor("#b9c7d8" if arm in gate_labels else "#dddddd")
        if row == len(body):
            cell.set_text_props(fontweight="bold")

    note = ("H− through = attacks the defense let run (lower is better)    "
            "H+ done = required tasks completed (higher is better)\n"
            "Grey headers: the paper's conditions. Blue headers: this project's gate. "
            "Memory off is 0/0 in every category and is not shown.")
    if nulls:
        note += (f"\n{len(nulls)} pairs failed the null control: their H− side is not evidence "
                 "about authority")
    fig.text(0.5, 0.03, note, ha="center", va="bottom", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"  written {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--module", default="both", help="b, c or both")
    parser.add_argument("--logs", default=LOGS_ROOT)
    args = parser.parse_args(argv)

    root = Path(args.logs)
    if not root.exists():
        print(f"  no {root}")
        return 1
    made = 0
    modules = ["b", "c"] if args.module == "both" else [args.module]
    for module in modules:
        args.module = module
        made += _draw_module(root, args)
    if not made:
        print("  nothing to draw yet — no episodes recorded for that module")
    return 0


def _draw_module(root: Path, args) -> int:
    made = 0
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for suffix, dataset in (("", "dataset 1 (dev)"), ("__heldout", "dataset 2 (held-out)")):
            data, nulls = collect(model_dir, args.module, suffix)
            if not data:
                continue
            title = f"Module {args.module.upper()} — {dataset}\n{model_dir.name}"
            name = f"module_{args.module}_{'heldout' if suffix else 'dev'}_{model_dir.name}.png"
            draw(data, nulls, title, OUT_DIR / name, args.module)
            made += 1
    return made


if __name__ == "__main__":
    sys.exit(main())
