"""
The selection semantics: the agent may choose among the customer's own values (new_src/bench/gate.py
since the agent's arguments became proposals).

Permission for (a, object o, arguments θ): some outstanding request for a; o is among the objects
the requests name, or, if none names one, among the customer's values under the object key; the
requests compatible with o are those naming o or no object; for each parameter, θ(p) is among the
values those requests name for it, or, if they name none, among the customer's values under its key.

Implemented literally from a history and as a canonical state machine, with the same checks as
authority.py: Nerode classes (Moore) and the greatest safe-replacement relation.

    python selection.py [N] [--revocable]
"""

import itertools
import sys
from collections import defaultdict

import numpy as np

import authority as A

REVOCABLE = "--revocable" in sys.argv
EVENTS = A.events("revocable" if REVOCABLE else "static")


def raw_permitted(history, op):
    a, o, args = op
    obj_key, ps = A.ACTIONS[a]
    vals = {k: set() for k in A.KEYS}
    out = {x: [] for x in A.ACTIONS}
    for e in history:
        kind = e[0]
        if kind == "assert":
            vals[e[1]].add(e[2])
        elif kind == "retract":
            vals[e[1]].discard(e[2])
        elif kind == "expire":
            vals[e[1]].clear()
        elif kind == "req":
            _, ra, ro, named = e
            rk, rps = A.ACTIONS[ra]
            if ro is not None:
                vals[rk].add(ro)
            for key, u in zip(rps.values(), named):
                if u is not None:
                    vals[key].add(u)
            out[ra].append((ro, named))
        elif kind == "withdraw":
            out[e[1]] = []
        elif kind == "exec":
            _, ea, eo, _ = e
            out[ea] = [r for r in out[ea] if not (r[0] is None or r[0] == eo)]
    reqs = out[a]
    if not reqs:
        return False
    if obj_key:
        named_objs = {r[0] for r in reqs if r[0] is not None}
        pool = named_objs or vals[obj_key]
        if o not in pool:
            return False
        reqs = [r for r in reqs if r[0] in (None, o)]
        if not reqs:
            return False
    for i, key in enumerate(ps.values()):
        named = {r[1][i] for r in reqs if r[1][i] is not None}
        if args[i] not in (named or vals[key]):
            return False
    return True


# canonical: keys are sets; a request state is a frozenset of (group, (named set per parameter))
def initial():
    return (tuple(frozenset() for _ in A.KEYS), tuple(frozenset() for _ in A.ACTIONS))


def _groups(r):
    return dict(r)


def permitted(s, op):
    keys, rho = s
    a, o, args = op
    obj_key, ps = A.ACTIONS[a]
    groups = _groups(rho[list(A.ACTIONS).index(a)])
    if not groups:
        return False
    if obj_key:
        named_objs = {g for g in groups if g is not None}
        pool = named_objs or keys[A.KEYS.index(obj_key)]
        if o not in pool:
            return False
        use = [groups[g] for g in (None, o) if g in groups]
        if not use:
            return False
    else:
        use = [groups[None]]
    for i, key in enumerate(ps.values()):
        named = set().union(*(g[i] for g in use))
        if args[i] not in (named or keys[A.KEYS.index(key)]):
            return False
    return True


def step(s, e):
    keys, rho = list(s[0]), list(s[1])
    kind = e[0]

    def add(k, v):
        i = A.KEYS.index(k)
        keys[i] = keys[i] | {v}

    if kind == "assert":
        add(e[1], e[2])
    elif kind == "retract":
        i = A.KEYS.index(e[1]); keys[i] = keys[i] - {e[2]}
    elif kind == "expire":
        keys[A.KEYS.index(e[1])] = frozenset()
    elif kind == "req":
        _, a, ro, named = e
        obj_key, ps = A.ACTIONS[a]
        if ro is not None:
            add(obj_key, ro)
        for key, u in zip(ps.values(), named):
            if u is not None:
                add(key, u)
        j = list(A.ACTIONS).index(a)
        groups = _groups(rho[j])
        cur = groups.get(ro, tuple(frozenset() for _ in ps))
        groups[ro] = tuple(c | ({u} if u is not None else set()) for c, u in zip(cur, named))
        rho[j] = frozenset(groups.items())
    elif kind == "withdraw":
        rho[list(A.ACTIONS).index(e[1])] = frozenset()
    elif kind == "exec":
        op = e[1:]
        if not permitted(s, op):
            return None
        j = list(A.ACTIONS).index(op[0])
        groups = _groups(rho[j])
        for g in (None, op[1]):
            groups.pop(g, None)
        rho[j] = frozenset(groups.items())
    return (tuple(keys), tuple(rho))


def closure(seeds):
    seen, order, frontier = set(seeds), list(seeds), list(seeds)
    while frontier:
        nxt = []
        for st in frontier:
            for e in EVENTS:
                t = step(st, e)
                if t is not None and t not in seen:
                    seen.add(t); order.append(t); nxt.append(t)
        frontier = nxt
    return order


def perm_set(s):
    return frozenset(op for op in A.OPS if permitted(s, op))


def check_raw(samples=3000, length=9, seed=0):
    import random
    rng = random.Random(seed)
    non_exec = [e for e in EVENTS if e[0] != "exec"]
    n = 0
    for hist_len in range(3):
        for hist in itertools.product(non_exec, repeat=hist_len):
            s = initial(); prefix = []
            for e in hist:
                s = step(s, e); prefix.append(e)
            for op in A.OPS:
                assert permitted(s, op) == raw_permitted(prefix, op), (prefix, op)
            n += 1
    for _ in range(samples):
        s = initial(); prefix = []
        for _ in range(length):
            if rng.random() < 0.25 and perm_set(s):
                e = ("exec",) + rng.choice(sorted(perm_set(s), key=repr))
            else:
                e = rng.choice(non_exec)
            s = step(s, e); prefix.append(e)
            for op in A.OPS:
                assert permitted(s, op) == raw_permitted(prefix, op), (prefix, op)
            n += 1
    return n


def moore(states):
    index = {s: i for i, s in enumerate(states)}
    trans = np.array([[index.get(step(s, e), -1) if step(s, e) is not None else -1 for e in EVENTS] for s in states])
    ids = {}
    cls = np.array([ids.setdefault(perm_set(s), len(ids)) for s in states])
    while True:
        sig = {}
        new = np.array([sig.setdefault((cls[i], tuple(cls[t] if t >= 0 else -1 for t in trans[i])), len(sig))
                        for i in range(len(states))])
        if len(sig) == len(set(cls.tolist())):
            return new
        cls = new


def safe_relation(mem, true):
    mi = {s: i for i, s in enumerate(mem)}; ti = {s: i for i, s in enumerate(true)}
    pm = [perm_set(s) for s in mem]; pt = [perm_set(s) for s in true]
    R = np.array([[pm[i] <= pt[j] for j in range(len(true))] for i in range(len(mem))])
    TM = np.array([[mi[t] if (t := step(s, e)) is not None else -1 for s in mem] for e in EVENTS])
    TT = np.array([[ti[t] if (t := step(s, e)) is not None else -1 for s in true] for e in EVENTS])
    rm, rt = np.arange(len(mem)), np.arange(len(true))
    while True:
        before = int(R.sum())
        for ei, e in enumerate(EVENTS):
            if e[0] == "exec":
                can = TM[ei] >= 0
                R &= (~can)[:, None] | R[np.ix_(np.where(can, TM[ei], rm), np.where(TT[ei] >= 0, TT[ei], rt))]
            else:
                R &= R[np.ix_(TM[ei], TT[ei])]
        if int(R.sum()) == before:
            return R


def reduced(s):
    """What the exact-memory theorem predicts for the selection semantics: the canonical state,
    with a request that names nothing dropped when a request naming an object is outstanding
    for the same action (it adds no name, and every execution spends it)."""
    keys, rho = s
    out = []
    for a, r in zip(A.ACTIONS, rho):
        groups = _groups(r)
        bare = groups.get(None)
        if bare is not None and not any(bare) and any(g is not None for g in groups):
            groups.pop(None)
        out.append(frozenset(groups.items()))
    return (keys, tuple(out))


if __name__ == "__main__":
    print(f"selection semantics, N = {A.N}, {'revocable' if REVOCABLE else 'static'}")
    print("raw == canonical on", check_raw(), "prefixes")
    states = closure([initial()])
    cls = moore(states)
    print(f"{len(states)} reachable canonical states, {len(set(cls.tolist()))} Nerode classes")
    groups = defaultdict(list)
    for s, c in zip(states, cls):
        groups[c].append(s)
    # The theorem assumes infinitely many values: a separating future may need a value nothing has
    # used yet. At finite N, states that have used every value can merge for lack of one, so the
    # comparison is made on states that leave at least FRESH values unused.
    fresh_needed = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--fresh=")), "1"))

    def used(st):
        keys, rho = st
        vals = set().union(*keys)
        for r in rho:
            for g, named in r:
                if g is not None:
                    vals.add(g)
                for n in named:
                    vals |= n
        return vals

    keep = [i for i, st in enumerate(states) if len(set(A.VALUES) - used(st)) >= fresh_needed]
    by_pred = defaultdict(set)
    for i in keep:
        by_pred[reduced(states[i])].add(cls[i])
    groups = defaultdict(list)
    for i in keep:
        groups[cls[i]].append(states[i])
    print(f"   comparing {len(keep)} states with at least {fresh_needed} unused value(s)")
    split = [p for p, cs in by_pred.items() if len(cs) > 1]
    merged = [g for g in groups.values() if len({reduced(s) for s in g}) > 1]
    ok = not split and not merged
    print(f"{'PASS' if ok else 'FAIL'}  exact memory (selection): {len(set(cls.tolist()))} classes, "
          f"{len(by_pred)} predicted; {len(split)} split, {len(merged)} merged")
    for g in merged[:4]:
        print("  equivalent but predicted different:", [reduced(s) for s in g][:3])
