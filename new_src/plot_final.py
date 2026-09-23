"""
Tables for the final run (logs_final), one image per module.

    python -m new_src.plot_final              # every model recorded under logs_final/
    python -m new_src.plot_final --model MiniMaxAI/MiniMax-M2.7

Rows are conditions, columns are suites, and each cell holds three lines:

    H−  k/n        attacks that got through, out of the H- episodes recorded
    H+  k/n        tasks performed, out of the H+ episodes recorded
    ASR x% · TSR y%

ASR excludes pairs whose null control fired — with the contested record removed the action
happened anyway, so that pair measures obedience to the closing request rather than authority.
TSR keeps every pair: the null control says nothing about the H+ side.

Besides the overview, one table per module and suite breaks the same counts down by attack
type: each cell reads "0/5 | 5/5" — attacks through out of the H- pairs, tasks performed out of
the H+ pairs — with the two rates underneath.

Images go to logs_result/, beside the earlier figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from new_src import final  # noqa: E402

OUT_DIR = Path("logs_result")

#: Column order. A suite appears only if something was recorded for it.
SUITES = [("core", "run", "core (dev)"), ("core", "heldout", "core (held-out)"),
          ("multiarg", "run", "multi-arg (dev)"), ("multiarg", "heldout", "multi-arg (held-out)"),
          ("speechact", "run", "speech-act (dev)"), ("speechact2", "heldout", "speech-act (held-out)"),
          ("licence", "run", "licence")]

#: Row order per module, by condition key, with the label to print.
ROWS = {
    "b": [("memory-off", "memory off"), ("baseline", "baseline (washed)"),
          ("baseline-attributed", "baseline (attributed)"), ("sanitizer", "sanitizer"),
          ("conservative-join", "conservative join"), ("gold-washed", "gold labels, washed"),
          ("gold-prompted", "prompted with labels"), ("gate", "gate (gold labels)"),
          ("gate-license-model", "gate + channels"),
          ("baseline-retrieve", "baseline, agent searches memory"),
          ("baseline-attributed-retrieve", "baseline (attributed), agent searches memory"),
          ("gate-retrieve", "gate + channels, agent searches memory")],
    "c": [("memory-off", "memory off"), ("c-no-label", "no label"), ("c-naive-join", "naive join"),
          ("c-predicted", "predicted label"), ("c-oracle", "oracle label"),
          ("gate-license-model", "gate + channels")],
}
GATE_ROWS = {"gate", "gate-license-model", "gate-retrieve"}


def _cells(model: str) -> tuple[dict, dict]:
    """{module: {(condition, suite): (through, n_minus, done, n_plus)}} and the fired null pairs."""
    fired: dict[tuple[str, str], set[str]] = {}
    data: dict[str, dict] = {"b": {}, "c": {}}
    for phase in final.PHASES:
        rows = final._rows(final.records_path(phase, model))
        if not rows:
            continue
        if phase.module == "null":
            fired[(phase.runner, phase.suite)] = {r["pair_id"] for r in rows
                                                  if r["variant"] == "H-" and r.get("performed")}
        elif phase.module in ("b", "c"):
            data[phase.module][(phase.condition, phase.suite, phase.runner)] = rows
    return data, fired


def _counts(rows: list[dict], excluded: set[str]) -> tuple[int, int, int, int]:
    minus = [r for r in rows if r["variant"] == "H-" and r["pair_id"] not in excluded]
    plus = [r for r in rows if r["variant"] == "H+"]
    return (sum(bool(r["performed"]) for r in minus), len(minus),
            sum(bool(r["performed"]) for r in plus), len(plus))


def draw(module: str, model: str, data: dict, fired: dict, path: Path) -> Path | None:
    present = [(suite, runner, label) for suite, runner, label in SUITES
               if any(key[1] == suite and key[2] == runner for key in data[module])]
    rows = [(key, label) for key, label in ROWS[module]
            if any(k[0] == key for k in data[module])]
    if not present or not rows:
        return None

    header = ["condition", *(label for _, _, label in present)]
    body, colours = [], []
    for key, label in rows:
        line = [label]
        tint = ["#eeeeee" if key in GATE_ROWS else "#f5f5f5"]
        for suite, runner, _ in present:
            recorded = data[module].get((key, suite, runner))
            if not recorded:
                line.append("—")
                tint.append("#ffffff")
                continue
            through, n_minus, done, n_plus = _counts(recorded, fired.get((runner, suite), set()))
            asr = f"{100 * through / n_minus:.0f}%" if n_minus else "—"
            tsr = f"{100 * done / n_plus:.0f}%" if n_plus else "—"
            line.append(f"H−  {through}/{n_minus}\nH+  {done}/{n_plus}\nASR {asr} · TSR {tsr}")
            # Red where attacks got through, green where none did and the tasks were done.
            tint.append("#f8d7da" if through else ("#d4edda" if n_minus and done == n_plus else "#e8f4ea"))
        body.append(line)
        colours.append(tint)

    fig, ax = plt.subplots(figsize=(3.2 + 2.5 * len(present), 0.9 + 0.62 * (len(rows) + 1)))
    ax.axis("off")
    ax.set_title(f"Module {module.upper()} — {model}\nH− attacks through · H+ tasks performed",
                 fontsize=13, fontweight="bold", pad=16)
    table = ax.table(cellText=body, colLabels=header, cellColours=colours, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.6)
    for (row, column), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#dfe6ee")
        elif column == 0:
            cell.set_text_props(fontweight="bold" if rows[row - 1][0] in GATE_ROWS else "normal",
                                ha="left")
        cell.set_edgecolor("#bbbbbb")
    fig.text(0.01, 0.01, "ASR excludes pairs whose null control fired; TSR keeps every pair.",
             fontsize=8, color="#555555")
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_by_category(module: str, suite: str, runner: str, suite_label: str, model: str,
                     data: dict, fired: dict, path: Path) -> Path | None:
    """One table per module and suite: conditions down the side, attack types across the top."""
    recorded = {key: rows for key, rows in data[module].items() if key[1] == suite and key[2] == runner}
    if not recorded:
        return None
    rows = [(key, label) for key, label in ROWS[module] if any(k[0] == key for k in recorded)]
    categories = sorted({r["category"] for rs in recorded.values() for r in rs})
    excluded = fired.get((runner, suite), set())

    header = ["condition", *categories, "all"]
    body, colours = [], []
    for key, label in rows:
        line = [label]
        tint = ["#eeeeee" if key in GATE_ROWS else "#f5f5f5"]
        for category in [*categories, None]:
            episodes = [r for r in recorded[(key, suite, runner)]
                        if category is None or r["category"] == category]
            through, n_minus, done, n_plus = _counts(episodes, excluded)
            asr = f"{100 * through / n_minus:.0f}%" if n_minus else "—"
            tsr = f"{100 * done / n_plus:.0f}%" if n_plus else "—"
            line.append(f"{through}/{n_minus} | {done}/{n_plus}\n{asr} | {tsr}")
            tint.append("#f8d7da" if through else ("#d4edda" if n_minus and done == n_plus else "#e8f4ea"))
        body.append(line)
        colours.append(tint)

    fig, ax = plt.subplots(figsize=(3.4 + 1.35 * (len(categories) + 1), 0.9 + 0.55 * (len(rows) + 1)))
    ax.axis("off")
    ax.set_title(f"Module {module.upper()} — {suite_label} — {model}\n"
                 "per attack type:  attacks through / H−   |   tasks performed / H+",
                 fontsize=12, fontweight="bold", pad=14)
    table = ax.table(cellText=body, colLabels=header, cellColours=colours, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 2.2)
    for (row, column), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#dfe6ee")
        elif column == 0:
            cell.set_text_props(fontweight="bold" if rows[row - 1][0] in GATE_ROWS else "normal", ha="left")
        elif column == len(header) - 1:
            cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("#bbbbbb")
    fig.text(0.01, 0.01, "ASR excludes pairs whose null control fired; TSR keeps every pair.",
             fontsize=8, color="#555555")
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# One picture per defense, per dataset, per model — every attack type in it
# ---------------------------------------------------------------------------

from new_src.plot_summary import ASR_COLOUR, GRID, INK, INK_2, SURFACE, TSR_COLOUR  # noqa: E402

#: (suite for the dev dataset, suite for the held-out dataset, category code, label). The
#: multi-argument types share the core transition codes, so they are named by their action; of the
#: speech-act families only the two that are not about provenance are shown — a customer quoting
#: someone else, and a grant arriving through a tool.
TYPES = [
    ("core", "core", "R2F", "R2F  report → fact"),
    ("core", "core", "P2R", "P2R  procedure → rule"),
    ("core", "core", "C2O", "C2O  claim → operational"),
    ("core", "core", "MIX", "MIX  mixed evidence"),
    ("core", "core", "O2I", "O2I  observation → intention"),
    ("core", "core", "R2P", "R2P  recommendation → preference"),
    ("core", "core", "S2D", "S2D  suggestion → decision"),
    ("multiarg", "multiarg", "C2O", "WIRE  multi-arg, 5 arguments"),
    ("multiarg", "multiarg", "P2R", "STO  multi-arg, 4 arguments"),
    ("multiarg", "multiarg", "O2I", "TRV  multi-arg, country as value"),
    ("speechact", "speechact2", "Q2D", "Q2D  customer quotes somebody"),
    ("speechact", "speechact2", "G2O", "G2O  grant through a tool"),
]

CONDITION_LABELS = {key: label for module in ROWS.values() for key, label in module}


def draw_condition(module: str, condition: str, dataset: str, model: str, data: dict, fired: dict,
                   path: Path) -> Path | None:
    """One defense (or baseline), one dataset, every attack type — as a table."""
    runner = "run" if dataset == "dev" else "heldout"
    rows = []
    for dev_suite, heldout_suite, category, label in TYPES:
        suite = dev_suite if dataset == "dev" else heldout_suite
        recorded = data[module].get((condition, suite, runner))
        if not recorded:
            continue
        episodes = [r for r in recorded if r["category"] == category]
        if episodes:
            rows.append((label, _counts(episodes, fired.get((runner, suite), set()))))
    if not rows:
        return None

    header = ["attack type", "H−  through", "ASR", "H+  done", "TSR"]
    body, colours = [], []
    for label, (through, n_minus, done, n_plus) in rows:
        body.append([label, f"{through}/{n_minus}", f"{100 * through / n_minus:.0f}%" if n_minus else "—",
                     f"{done}/{n_plus}", f"{100 * done / n_plus:.0f}%" if n_plus else "—"])
        attack = "#f8d7da" if through else "#d4edda"
        task = "#d4edda" if n_plus and done == n_plus else ("#fff3cd" if n_plus else "#ffffff")
        colours.append(["#f5f5f5", attack, attack, task, task])

    through = sum(r[1][0] for r in rows); n_minus = sum(r[1][1] for r in rows)
    done = sum(r[1][2] for r in rows); n_plus = sum(r[1][3] for r in rows)
    body.append(["all types", f"{through}/{n_minus}", f"{100 * through / max(n_minus, 1):.0f}%",
                 f"{done}/{n_plus}", f"{100 * done / max(n_plus, 1):.0f}%"])
    colours.append(["#e4e4e4"] * 5)

    fig, ax = plt.subplots(figsize=(8.4, 1.5 + 0.42 * (len(body) + 1)))
    ax.axis("off")
    label = CONDITION_LABELS.get(condition, condition)
    ax.set_title(f"Module {module.upper()} — {label} — "
                 f"{'development set' if dataset == 'dev' else 'held-out set'}\n{model}",
                 fontsize=12, fontweight="bold", pad=14)
    table = ax.table(cellText=body, colLabels=header, cellColours=colours, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.9)
    # The type names are long; give the first column the room it needs instead of clipping them.
    widths = [0.40, 0.15, 0.10, 0.15, 0.10]
    for (row, column), cell in table.get_celld().items():
        cell.set_width(widths[column])
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#dfe6ee")
        else:
            if column == 0:
                cell.set_text_props(ha="left")
                cell.PAD = 0.03
            if row == len(body):
                cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("#bbbbbb")
    fig.text(0.01, 0.01, "ASR excludes pairs whose null control fired. Five pairs per type, so a single "
                         "type carries no conclusion on its own.", fontsize=8, color="#555555")
    fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="default: the model of the recorded run")
    args = parser.parse_args(argv)

    models = [args.model] if args.model else ([json.loads(final.CONFIG.read_text())["model"]]
                                              if final.CONFIG.exists() else [])
    if not models:
        print("  No final run recorded; nothing to draw.")
        return 1
    for model in models:
        data, fired = _cells(model)
        slug = model.replace("/", "_").replace(":", "_")
        for module in ("b", "c"):
            path = draw(module, model, data, fired, OUT_DIR / f"final_module_{module}_{slug}.png")
            print(f"  {path}" if path else f"  Module {module.upper()}: nothing recorded yet")
            for dataset in ("dev", "heldout"):
                for condition, _ in ROWS[module]:
                    name = f"final_{module}_{dataset}_{condition}_{slug}.png"
                    if (drawn := draw_condition(module, condition, dataset, model, data, fired,
                                                OUT_DIR / name)):
                        print(f"  {drawn}")
            for suite, runner, suite_label in SUITES:
                name = f"final_module_{module}_{suite}_{'heldout' if runner == 'heldout' else 'dev'}_{slug}.png"
                if (drawn := draw_by_category(module, suite, runner, suite_label, model, data, fired,
                                              OUT_DIR / name)):
                    print(f"  {drawn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
