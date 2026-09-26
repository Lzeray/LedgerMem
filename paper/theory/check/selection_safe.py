"""Safe forgetting in the selection semantics: the predicted relation against the computed one.

Prediction (a memory state m in place of the truth t):
  * every key holds a subset of the truth's values;
  * every request group of m exists in t and names a subset of what t's names, and never an empty
    set where t's is not;
  * every request that names an object or a value is kept; only one naming nothing may be dropped.
In words: forget values freely, but never let a narrowing request fall back to the wider key.

    python selection_safe.py [N] [--one-action]
"""
import sys
from collections import defaultdict

import numpy as np

import authority as A
import selection as S


def predict(m, t):
    """m safe in place of t: forget values freely, never lose what narrows a choice."""
    for i in range(len(A.KEYS)):
        if not m[0][i] <= t[0][i]:
            return False
    for a in A.ACTIONS:
        j = list(A.ACTIONS).index(a)
        gm, gt = dict(m[1][j]), dict(t[1][j])
        for g, named in gm.items():
            # nothing invented: a group, and each name in it, must exist in the truth
            if g not in gt or not all(x <= y for x, y in zip(named, gt[g])):
                return False
            # a group kept must keep narrowing every parameter it narrows in the truth
            if any(y and not x for x, y in zip(named, gt[g])):
                return False
        # every request that narrows anything is kept: one naming an object narrows which object
        # a later request may concern, even after the others are spent, and one naming a value
        # narrows every later request merged into it. Only a request naming nothing may go.
        for g, named in gt.items():
            if (g is not None or any(named)) and g not in gm:
                return False
    return True


if __name__ == "__main__":
    states = S.closure([S.initial()])
    print(f"selection safe forgetting, N = {A.N}, {len(states)} states")
    cls = S.moore(states)
    R = S.safe_relation(states, states)
    P = np.array([[predict(m, t) for t in states] for m in states])
    unsound = np.argwhere(P & ~R)
    missed = np.argwhere(R & ~P)
    groups = defaultdict(list)
    for i, c in enumerate(cls):
        groups[c].append(i)
    def fresh(st):
        vals = set().union(*st[0])
        for r in st[1]:
            for g, named in r:
                vals |= ({g} if g is not None else set()) | set().union(*named)
        return len(set(A.VALUES) - vals) >= 1

    # as in the exact check: at finite N a state that has used every value can look safe for lack
    # of a fresh value to separate it; only pairs leaving one unused are compared
    unexplained = [(i, j) for i, j in missed if fresh(states[i]) and fresh(states[j])
                   and not any(P[i2, j2] for i2 in groups[cls[i]] for j2 in groups[cls[j]])]
    ok = not len(unsound) and not unexplained
    print(f"{'PASS' if ok else 'FAIL'}  {int(R.sum())} safe pairs; {len(unsound)} predicted-safe but unsafe; "
          f"{len(unexplained)} safe but not predicted (after equivalence)")
    for i, j in [tuple(x) for x in unsound[:3]]:
        print("  predicted safe, unsafe:  mem", states[i], "\n                         true", states[j])
    for i, j in unexplained[:3]:
        print("  safe, not predicted:     mem", states[i], "\n                         true", states[j])
