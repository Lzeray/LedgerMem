"""Lemma (one register), checked by computing the greatest safe relation on all register states.

    python register_check.py [N]
"""
import itertools
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
V = tuple(range(1, N + 1))
MARK = "◇"


def read(s):
    return next(iter(s)) if len(s) == 1 and MARK not in s else None


def check(revocable):
    atoms = V + ((MARK,) if revocable else ())
    states = [frozenset(c) for r in range(len(atoms) + 1) for c in itertools.combinations(atoms, r)]
    if not revocable:  # static: flat, only reachable shapes; sets of two or more are one state
        states = [frozenset()] + [frozenset({v}) for v in V] + [frozenset(V)]

    def norm(s):
        return s if revocable or len(s) <= 1 else frozenset(V)

    events = [("add", v) for v in V] + ([("retract", v) for v in V] + [("clear",)] if revocable else [])

    def step(s, e):
        if e[0] == "add":
            return norm(s | {e[1]})
        if e[0] == "retract":
            return s - {e[1]}
        return frozenset()

    R = {(a, b) for a in states for b in states if read(a) is None or read(a) == read(b)}
    while True:
        R2 = {(a, b) for (a, b) in R if all((step(a, e), step(b, e)) in R for e in events)}
        if R2 == R:
            break
        R = R2
    if revocable:
        pred = {(a, b) for a in states for b in states if a == b or MARK in a}
    else:
        pred = {(a, b) for a in states for b in states if a == b or len(a) >= 2}
    return R == pred, len(R), len(states)


for rev in (False, True):
    ok, n, m = check(rev)
    print(f"{'PASS' if ok else 'FAIL'}  register lemma ({'revocable' if rev else 'static'}), N = {N}: "
          f"{n} safe pairs over {m} states")
