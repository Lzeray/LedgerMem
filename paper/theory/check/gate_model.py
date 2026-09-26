"""
The gate's steps 1-2 (new_src/bench/gate.py, no customer attached) on records, checked against the
permission semantics by enumeration.

A history is a sequence of messages. Each message is the customer's own words, the customer quoting
someone, or an outside source; its content is an assertion or a request (a request asserts what it
names). A faithful write stores:
    own words   -> authorized with its requests, or LOWERED (below authorized, requests lost,
                   slots kept): the misread "quotation" of the customer's own words;
    a quotation -> unendorsed, on the customer's channel, no requests;
    outside     -> unendorsed, another channel, no requests.
Checked: whenever the gate executes, the permission semantics permits (Theorem 1 with the lowering
proposition). Also checked: without the veto on lowered customer words, a counterexample exists.

    python gate_model.py [N] [max_len]
"""

import itertools
import sys

import authority as A

MAX_LEN = int(sys.argv[2]) if len(sys.argv) > 2 else 3


def contents():
    out = [("assert", k, v) for k in A.KEYS for v in A.VALUES]
    out += [("req",) + r for a in A.ACTIONS for r in A.requests_for(a)]
    return out


def slots_of(c):
    if c[0] == "assert":
        return {(c[1], c[2])}
    _, a, o, named = c
    obj_key, ps = A.ACTIONS[a]
    s = {(obj_key, o)} if o is not None else set()
    return s | {(key, u) for key, u in zip(ps.values(), named) if u is not None}


def gate_executes(records, op, veto=True):
    """records: list of (label, channel, slots, request or None)."""
    a, o, args = op
    obj_key, ps = A.ACTIONS[a]
    C = [r for r in records if r[0] == "Au" and r[3] is not None and r[3][0] == a]
    if not C:
        return False

    def lowered(key):
        return {v for (lab, ch, sl, _) in records if lab != "Au" and ch == "usr" for (k, v) in sl if k == key}

    scope = None
    if obj_key:
        named = {r[3][1] for r in C if r[3][1] is not None}
        if len(named) > 1:
            return False
        if named:
            scope = next(iter(named))
        else:
            vals = {v for (lab, ch, sl, _) in records if lab == "Au" for (k, v) in sl if k == obj_key}
            if len(vals) != 1:
                return False
            scope = next(iter(vals))
        if veto and lowered(obj_key) - {scope}:
            return False
        if o != scope:
            return False
    lam = [r for r in C if r[3][1] in (None, scope)]
    for i, key in enumerate(ps.values()):
        named = {r[3][2][i] for r in lam if r[3][2][i] is not None}
        if len(named) > 1:
            return False
        if named:
            u = next(iter(named))
            if veto and lowered(key) - {u}:
                return False
            bound = u
        else:
            rows = [(lab, v) for (lab, ch, sl, _) in records for (k, v) in sl if k == key]
            vals = {v for _, v in rows}
            if len(vals) != 1 or not any(lab == "Au" for lab, _ in rows):
                return False
            bound = next(iter(vals))
        if args[i] != bound:
            return False
    return True


def run_dropped():
    """Completeness is necessary: drop one own-voice message entirely (not written at all) and look
    for an unsafe execution. The theory predicts one exists."""
    kinds = ("own", "quote", "outside")
    cs = contents()
    for n in range(1, MAX_LEN + 1):
        for msgs in itertools.product(itertools.product(kinds, cs), repeat=n):
            history = [c for k, c in msgs if k == "own"]
            truth = {op for op in A.OPS if A.raw_permitted(history, op)}
            for drop in [i for i, (k, _) in enumerate(msgs) if k == "own"]:
                records = []
                for i, (k, c) in enumerate(msgs):
                    if i == drop:
                        continue
                    if k == "own":
                        records.append(("Au", "usr", slots_of(c), c[1:] if c[0] == "req" else None))
                    elif k == "quote":
                        records.append(("Un", "usr", slots_of(c), None))
                    else:
                        records.append(("Un", "ut", slots_of(c), None))
                for op in A.OPS:
                    if gate_executes(records, op) and op not in truth:
                        return msgs, drop, op
    return None


def run(veto):
    kinds = ("own", "quote", "outside")
    cs = contents()
    for n in range(1, MAX_LEN + 1):
        for msgs in itertools.product(itertools.product(kinds, cs), repeat=n):
            own_idx = [i for i, (k, _) in enumerate(msgs) if k == "own"]
            history = [c for k, c in msgs if k == "own"]
            truth = {op for op in A.OPS if A.raw_permitted(history, op)}
            for lowered_mask in itertools.product((False, True), repeat=len(own_idx)):
                low = {i for i, m in zip(own_idx, lowered_mask) if m}
                records = []
                for i, (k, c) in enumerate(msgs):
                    if k == "own" and i not in low:
                        records.append(("Au", "usr", slots_of(c), c[1:] if c[0] == "req" else None))
                    elif k in ("own", "quote"):
                        records.append(("Un", "usr", slots_of(c), None))
                    else:
                        records.append(("Un", "ut", slots_of(c), None))
                for op in A.OPS:
                    if gate_executes(records, op, veto) and op not in truth:
                        return msgs, sorted(low), op
    return None


if __name__ == "__main__":
    print(f"N = {A.N}, histories up to {MAX_LEN} messages")
    bad = run(veto=True)
    print("PASS  gate sound under faithful writes with any lowering (veto on)" if bad is None
          else f"FAIL  counterexample with veto: {bad}")
    bad = run(veto=False)
    print("veto needed: counterexample without it:", bad if bad else "none found")
    bad = run_dropped()
    print("completeness needed: counterexample with one customer message unwritten:", bad if bad else "none found")
