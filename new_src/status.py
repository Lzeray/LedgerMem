"""
How far along every recorded run is.

    python -m new_src.status

Reads the episode logs and prints, per suite and condition, how many episodes are recorded
out of how many the suite contains, and what is left. Nothing here calls a model or touches
the database, so it is safe to run while a sweep is going.
"""

from __future__ import annotations

import sys
from pathlib import Path

from new_src.bench.metrics import load_jsonl
from new_src.config import LOGS_ROOT


def _expected() -> dict[str, int]:
    """Episodes a complete run of each suite contains: pairs x 2 variants."""
    from new_src.data.heldout import HELDOUT_SUITE
    from new_src.data.license_attacks import LICENSE_ATTACKS
    from new_src.data.speech_act_attacks import SPEECH_ACT_PAIRS
    from new_src.data.suite import SUITE

    return {
        "": len(SUITE) * 2,                      # dataset 1, the development suite
        "__heldout": len(HELDOUT_SUITE) * 2,     # dataset 2, the held-out suite
        "__heldout_speechact": len(SPEECH_ACT_PAIRS) * 2,
        "__licence": len(LICENSE_ATTACKS) * 2,
    }


def _suite_of(condition_dir: str) -> str:
    for suffix in ("__heldout_speechact", "__heldout", "__licence"):
        if condition_dir.endswith(suffix):
            return suffix
    return ""


SUITE_NAME = {
    "": "dataset 1 (dev)",
    "__heldout": "dataset 2 (held-out)",
    "__heldout_speechact": "speech-act families",
    "__licence": "licence attacks",
}


def main() -> int:
    root = Path(LOGS_ROOT)
    if not root.exists():
        print(f"  no {root} yet")
        return 0
    expected = _expected()
    rows = []
    for path in sorted(root.rglob("episodes.jsonl")):
        parts = path.parts
        model, module, condition = parts[-4], parts[-3], parts[-2]
        records = load_jsonl(path)
        unique = {(r["pair_id"], r["variant"]) for r in records}
        suite = _suite_of(condition)
        total = expected.get(suite, 0)
        if module == "null_control" or condition.startswith("null_control"):
            # The null control runs H- only, so a complete pass is half a suite.
            total = expected.get(suite, 0) // 2
        rows.append((model, module, condition, suite, len(unique), len(records), total))

    if not rows:
        print("  nothing recorded yet")
        return 0

    width = max(len(r[2]) for r in rows)
    current = None
    for model, module, condition, suite, unique, total_rows, total in sorted(rows):
        if model != current:
            print(f"\n  {model}")
            current = model
        left = max(total - unique, 0) if total else 0
        bar = "done" if total and unique >= total else f"{left} left"
        dupes = f"  ({total_rows - unique} duplicate rows)" if total_rows > unique else ""
        print(f"    {module:10} {condition[:width]:{width}}  {unique:>3}/{total or '?':<4} {bar:>9}"
              f"   [{SUITE_NAME.get(suite, suite)}]{dupes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
