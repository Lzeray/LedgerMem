"""
The gate with agent proposals (new_src/bench/gate.py) against the selection semantics.

For every history of up to MAX_LEN messages, every pattern of lowered customer messages, and every
operation with every pattern of proposals (each argument proposed as the operation's value, or not
proposed), the gate's decision without a customer is compared with the selection permission.

    python gate_selection.py [N] [max_len]
"""
import itertools
import sys

import authority as A
import selection as S
from gate_model import contents, slots_of

MAX_LEN = int(sys.argv[2]) if len(sys.argv) > 2 else 3


def gate_executes(records, op, proposal):
    a, o, args = op
    obj_key, ps = A.ACTIONS[a]
    C = [r for r in records if r[0] == "Au" and r[3] is not None and r[3][0] == a]
    if not C:
        return False

    def lowered(key):
        return {v for (lab, ch, sl, _) in records if lab != "Au" and ch == "usr" for (k, v) in sl if k == key}

    def au_values(key):
        return {v for (lab, ch, sl, _) in records if lab == "Au" for (k, v) in sl if k == key}

    scope = None
    if obj_key:
        named = {r[3][1] for r in C if r[3][1] is not None}
        if proposal.get("object") is not None:
            pool = named or au_values(obj_key)
            if proposal["object"] not in pool:
                return False
            scope = proposal["object"]
        else:
            if len(named) > 1:
                return False
            if named:
                scope = next(iter(named))
            else:
                vals = au_values(obj_key)
                if len(vals) != 1:
                    return False
                scope = next(iter(vals))
        if lowered(obj_key) - {scope}:
            return False
        if o != scope:
            return False
    lam = [r for r in C if r[3][1] in (None, scope)]
    for i, key in enumerate(ps.values()):
        pv = proposal.get(i)
        named = {r[3][2][i] for r in lam if r[3][2][i] is not None}
        if named:
            if pv is not None:
                if pv not in named:
                    return False
                chosen = pv
            elif len(named) > 1:
                return False
            else:
                chosen = next(iter(named))
        else:
            if pv is not None:
                if pv not in au_values(key):
                    return False
                chosen = pv
            else:
                rows = [(lab, v) for (lab, ch, sl, _) in records for (k, v) in sl if k == key]
                vals = {v for _, v in rows}
                if len(vals) != 1 or not any(lab == "Au" for lab, _ in rows):
                    return False
                chosen = next(iter(vals))
        if lowered(key) - {chosen}:
            return False
        if args[i] != chosen:
            return False
    return True


def proposals(op):
    a, o, args = op
    obj_key, _ = A.ACTIONS[a]
    for po in ((None, o) if obj_key else (None,)):
        for pa in itertools.product(*[(None, v) for v in args]):
            prop = {"object": po}
            prop.update({i: v for i, v in enumerate(pa)})
            yield prop


def run(drop_mode=False):
    kinds = ("own", "quote", "outside")
    cs = contents()
    for n in range(1, MAX_LEN + 1):
        for msgs in itertools.product(itertools.product(kinds, cs), repeat=n):
            own_idx = [i for i, (k, _) in enumerate(msgs) if k == "own"]
            history = [c for k, c in msgs if k == "own"]
            truth = {op for op in A.OPS if S.raw_permitted(history, op)}
            patterns = ([("drop", i) for i in own_idx] if drop_mode else
                        [("lower", frozenset(i for i, m in zip(own_idx, mask) if m))
                         for mask in itertools.product((False, True), repeat=len(own_idx))])
            for kind, what in patterns:
                records = []
                for i, (k, c) in enumerate(msgs):
                    if kind == "drop" and i == what:
                        continue
                    if k == "own" and not (kind == "lower" and i in what):
                        records.append(("Au", "usr", slots_of(c), c[1:] if c[0] == "req" else None))
                    elif k in ("own", "quote"):
                        records.append(("Un", "usr", slots_of(c), None))
                    else:
                        records.append(("Un", "ut", slots_of(c), None))
                for op in A.OPS:
                    if op in truth:
                        continue
                    for prop in proposals(op):
                        if gate_executes(records, op, prop):
                            return msgs, kind, what, op, prop
    return None


if __name__ == "__main__":
    print(f"N = {A.N}, histories up to {MAX_LEN} messages, all proposal patterns")
    bad = run()
    print("PASS  gate with proposals sound for the selection semantics under any lowering" if bad is None
          else f"FAIL  {bad}")
    bad = run(drop_mode=True)
    print("one customer message unwritten:", "counterexample " + str(bad) if bad else "no counterexample")
