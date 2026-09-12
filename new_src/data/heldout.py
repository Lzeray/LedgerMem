"""
The held-out suite: 5 fresh base histories × the 7 established transitions = 35 pairs.

Written by a model that was shown `GENERATION_PROMPT.md` and nothing about the defense, after
the gate's design was frozen. The 35 pairs in `suite.py` are the development set — the audit,
MIX's rewording and the licence check were all designed against them, so a number measured
there is a training score. Headline numbers belong here instead.

The narrative material lives in `heldout_raw.json` exactly as the generator emitted it. Only
the builders in `builders.py` turn it into pairs, so the carrier-swap invariants are produced
by the same code that produces the development suite and cannot be broken by a rephrasing.
Nothing in this file edits the generated text.
"""

from __future__ import annotations

import json
from pathlib import Path

from new_src.bench.taxonomy import Category
from new_src.data.builders import Base, c2o, mix, o2i, p2r, r2f, r2p, s2d

RAW_PATH = Path(__file__).with_name("heldout_raw.json")

_BUILDERS = {"R2F": r2f, "P2R": p2r, "C2O": c2o, "MIX": mix, "O2I": o2i, "R2P": r2p, "S2D": s2d}


def _load() -> tuple[dict[str, Base], dict[str, list]]:
    raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    bases = {
        entry["base_id"]: Base(
            entry["base_id"], entry["customer_id"], entry["customer_name"], entry["identity_fact"]
        )
        for entry in raw["bases"]
    }
    built: dict[str, list] = {}
    for code, builder in _BUILDERS.items():
        pairs = []
        for entry in raw[code]:
            fields = {key: value for key, value in entry.items() if key != "base_id"}
            pairs.append(builder(bases[entry["base_id"]], **fields))
        built[code] = pairs
    return bases, built


BASES, PAIRS_BY_CATEGORY = _load()

R2F_PAIRS = PAIRS_BY_CATEGORY["R2F"]
P2R_PAIRS = PAIRS_BY_CATEGORY["P2R"]
C2O_PAIRS = PAIRS_BY_CATEGORY["C2O"]
MIX_PAIRS = PAIRS_BY_CATEGORY["MIX"]
O2I_PAIRS = PAIRS_BY_CATEGORY["O2I"]
R2P_PAIRS = PAIRS_BY_CATEGORY["R2P"]
S2D_PAIRS = PAIRS_BY_CATEGORY["S2D"]

#: Ordered exactly like `suite.SUITE` so the two are directly comparable pair for pair.
HELDOUT_SUITE = [
    *R2F_PAIRS, *P2R_PAIRS, *C2O_PAIRS, *MIX_PAIRS, *O2I_PAIRS, *R2P_PAIRS, *S2D_PAIRS,
]

BY_CATEGORY = {
    Category.R2F: R2F_PAIRS, Category.P2R: P2R_PAIRS, Category.C2O: C2O_PAIRS,
    Category.MIX: MIX_PAIRS, Category.O2I: O2I_PAIRS, Category.R2P: R2P_PAIRS,
    Category.S2D: S2D_PAIRS,
}
