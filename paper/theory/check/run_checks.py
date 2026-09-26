"""Run every check in authority.py and print a verdict per theorem.

    python run_checks.py            # static and revocable, N = 3 (safe/revocable at N = 2)
"""

import sys
import time
from collections import defaultdict

import numpy as np

import authority as A


def verdict(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}", flush=True)
    return ok


def check_exact(regime, predict):
    states = A.reachable(regime)
    cls, _ = A.moore_classes(regime, states)
    pred = [predict(s) for s in states]
    by_moore, by_pred = defaultdict(set), defaultdict(set)
    for i, s in enumerate(states):
        by_moore[cls[i]].add(pred[i])
        by_pred[pred[i]].add(cls[i])
    split = [p for p, cs in by_pred.items() if len(cs) > 1]   # predicted equal, really different
    merged = [c for c, ps in by_moore.items() if len(ps) > 1]  # predicted different, really equal
    ok = not split and not merged
    verdict(f"exact minimal memory ({regime}): {len(states)} reachable states, "
            f"{len(by_moore)} Nerode classes, {len(by_pred)} predicted", ok,
            "" if ok else f"{len(split)} predicted classes split, {len(merged)} true classes merged")
    if merged:
        c = merged[0]
        members = [s for i, s in enumerate(states) if cls[i] == c]
        print("   example of states equivalent but predicted different:")
        for s in members[:4]:
            print("     ", s, "->", predict(s))
    if split:
        p = split[0]
        members = [s for i, s in enumerate(states) if pred[i] == p]
        print("   example of states predicted equal but different:", members[:3])
    return ok, states, cls


def check_safe(regime, mem_states, true_states, predict, mem_cls=None, true_cls=None):
    t0 = time.time()
    R = A.safe_relation(regime, mem_states, true_states)
    P = np.array([[predict(m, t) for t in true_states] for m in mem_states])
    unsound = np.argwhere(P & ~R)       # predicted safe, is not: the theorem's "if" is wrong
    missed = np.argwhere(R & ~P)        # safe, not predicted: "only if" too strong (maybe equivalence)
    unexplained = []
    if len(missed) and mem_cls is not None:
        # a miss is explained if an equivalent pair is predicted safe
        mem_groups, true_groups = defaultdict(list), defaultdict(list)
        for i, c in enumerate(mem_cls):
            mem_groups[c].append(i)
        for j, c in enumerate(true_cls):
            true_groups[c].append(j)
        for i, j in missed:
            if not any(P[i2, j2] for i2 in mem_groups[mem_cls[i]] for j2 in true_groups[true_cls[j]]):
                unexplained.append((i, j))
    elif len(missed):
        unexplained = [tuple(x) for x in missed]
    ok = not len(unsound) and not unexplained
    verdict(f"safe forgetting ({regime}): {R.sum()} safe pairs of {R.size}, "
            f"{len(missed) - len(unexplained)} explained by equivalence", ok,
            "" if ok else f"{len(unsound)} predicted-safe pairs unsafe, {len(unexplained)} safe pairs not predicted")
    for i, j in [tuple(x) for x in unsound[:3]]:
        print("   predicted safe but unsafe: mem", mem_states[i], " true", true_states[j])
    for i, j in unexplained[:3]:
        print("   safe but not predicted:    mem", mem_states[i], " true", true_states[j])
    print(f"   ({time.time() - t0:.0f}s)")
    return ok


if __name__ == "__main__":
    print(f"values N = {A.N}")
    results = []
    for regime in ("static", "revocable"):
        n = A.check_raw_vs_canonical(regime, exhaustive_len=2, samples=4000, sample_len=10)
        results.append(verdict(f"raw semantics == canonical machine ({regime})", True, f"{n} prefixes"))
    ok, st_states, st_cls = check_exact("static", A.predicted_class_static)
    results.append(ok)
    ok, rv_states, rv_cls = check_exact("revocable", A.predicted_class_revocable)
    results.append(ok)
    if "--safe" in sys.argv:
        results.append(check_safe("static", st_states, st_states, A.predicted_safe_static, st_cls, st_cls))
    if "--safe" in sys.argv or "--safe-revocable" in sys.argv:
        mem = A.with_marks(rv_states)
        print(f"   revocable memory states with marks: {len(mem)}")
        results.append(check_safe("revocable", mem, rv_states, A.predicted_safe_revocable))
    print("ALL PASS" if all(results) else "SOME FAILED")
