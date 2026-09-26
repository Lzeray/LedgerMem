"""
Brute-force checks of the theorems in paper/theory/, on small instances.

Two semantics are modelled, each twice: once literally from the definitions (a history is a list
of events, and permission is computed from the whole list), and once as the canonical state
machine the theorems are about. The checks are:

  raw == canonical     permission computed both ways agrees on every history tried;
  exact                the Nerode classes of the canonical machine (Moore minimisation) are
                       exactly the classes the exact-memory theorem predicts;
  safe                 the greatest "safe replacement" relation (a system remembering s' in place
                       of the true s never executes what s forbids, in any future) is exactly the
                       relation the safe-forgetting theorem predicts.

The semantics:

  "static"   assertions, requests, permitted executions (paper/theory/00_setting.tex).
  "revocable" adds retraction of an asserted value, withdrawal of an action's requests, and
             expiry of a key (all of its values, markers included). Memory may also hold the
             marker MARK: a value nobody can retract, standing for "something was forgotten here".

Instance: keys ks, kp; action a (object read from ks, parameter p read from kp); action b (no
object, parameter q read from kp). Values 1..N.
"""

from __future__ import annotations

import itertools
import random
import sys
from dataclasses import dataclass

import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3
VALUES = tuple(range(1, N + 1))
KEYS = ("ks", "kp")
#: action -> (object key or None, {parameter: key})
ACTIONS = {"a": ("ks", {"p": "kp"}), "b": (None, {"q": "kp"})}
if "--one-action" in sys.argv:
    ACTIONS = {"a": ("ks", {"p": "kp"})}
BOT, TOP, MARK = "⊥", "⋆", "?"


def params(a):
    return tuple(ACTIONS[a][1])


def reads(a):
    obj_key, ps = ACTIONS[a]
    return tuple(([obj_key] if obj_key else []) + list(ps.values()))


def operations():
    """Every operation (a, object, arguments)."""
    out = []
    for a, (obj_key, ps) in ACTIONS.items():
        objs = VALUES if obj_key else (None,)
        for o in objs:
            for args in itertools.product(VALUES, repeat=len(ps)):
                out.append((a, o, args))
    return tuple(out)


OPS = operations()


def requests_for(a):
    obj_key, ps = ACTIONS[a]
    objs = (None,) + VALUES if obj_key else (None,)
    return [(a, o, named) for o in objs for named in itertools.product((None,) + VALUES, repeat=len(ps))]


def events(regime):
    ev = [("assert", k, v) for k in KEYS for v in VALUES]
    ev += [("req",) + r for a in ACTIONS for r in requests_for(a)]
    ev += [("exec",) + op for op in OPS]
    if regime == "revocable":
        ev += [("retract", k, v) for k in KEYS for v in VALUES]
        ev += [("withdraw", a) for a in ACTIONS]
        ev += [("expire", k) for k in KEYS]
    return ev


# --- the literal semantics: permission from the whole history ----------------------------------


def raw_permitted(history, op):
    a, o, args = op
    obj_key, ps = ACTIONS[a]
    vals = {k: set() for k in KEYS}
    out = {x: [] for x in ACTIONS}
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
            rk, rps = ACTIONS[ra]
            if ro is not None:
                vals[rk].add(ro)
            for (p, key), u in zip(rps.items(), named):
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

    def st(k):
        s = vals[k]
        return BOT if not s else (next(iter(s)) if len(s) == 1 else TOP)

    if obj_key:
        named_objs = {r[0] for r in reqs if r[0] is not None}
        if len(named_objs) > 1:
            return False
        o_star = next(iter(named_objs)) if named_objs else st(obj_key)
        if o_star in (BOT, TOP) or o != o_star:
            return False
    for i, (p, key) in enumerate(ps.items()):
        named = {r[1][i] for r in reqs if r[1][i] is not None}
        if len(named) > 1:
            return False
        want = next(iter(named)) if named else st(key)
        if want in (BOT, TOP) or args[i] != want:
            return False
    return True


# --- the canonical state machine -----------------------------------------------------------------
#
# A state is (keys, rho): keys maps each key to its canonical content, rho each action to
# "absent", "blocked", or (object or None, named tuple).
#   static:    a key is BOT, a value, or TOP (flat).
#   revocable: a key is a frozenset of values, possibly with MARK.


@dataclass(frozen=True)
class State:
    keys: tuple
    rho: tuple

    def key(self, k):
        return self.keys[KEYS.index(k)]

    def req(self, a):
        return self.rho[list(ACTIONS).index(a)]


def initial(regime):
    empty = BOT if regime == "static" else frozenset()
    return State(tuple(empty for _ in KEYS), tuple("absent" for _ in ACTIONS))


def _join_flat(s, v):
    return v if s == BOT else (s if s == v else TOP)


def key_reading(regime, s):
    """What a parameter sees when nothing is named: a value, or BOT / TOP."""
    if regime == "static":
        return s
    real = {v for v in s if v != MARK}
    if not s:
        return BOT
    if len(s) == 1 and real:
        return next(iter(real))
    return TOP


def _merge(r, ro, named):
    if r == "blocked":
        return "blocked"
    if r == "absent":
        return (ro, named)
    o, n = r
    if o is not None and ro is not None and o != ro:
        return "blocked"
    merged = []
    for x, y in zip(n, named):
        if x is not None and y is not None and x != y:
            return "blocked"
        merged.append(x if x is not None else y)
    return (o if o is not None else ro, tuple(merged))


def permitted(regime, s: State, op):
    a, o, args = op
    obj_key, ps = ACTIONS[a]
    r = s.req(a)
    if r in ("absent", "blocked"):
        return False
    ro, named = r
    if obj_key:
        want = ro if ro is not None else key_reading(regime, s.key(obj_key))
        if want in (BOT, TOP) or want != o:
            return False
    for i, key in enumerate(ps.values()):
        want = named[i] if named[i] is not None else key_reading(regime, s.key(key))
        if want in (BOT, TOP) or want != args[i]:
            return False
    return True


def perm_set(regime, s):
    return frozenset(op for op in OPS if permitted(regime, s, op))


def step(regime, s: State, e):
    keys, rho = list(s.keys), list(s.rho)
    kind = e[0]

    def add(k, v):
        i = KEYS.index(k)
        keys[i] = _join_flat(keys[i], v) if regime == "static" else keys[i] | {v}

    if kind == "assert":
        add(e[1], e[2])
    elif kind == "retract":
        i = KEYS.index(e[1])
        keys[i] = keys[i] - {e[2]}
    elif kind == "expire":
        keys[KEYS.index(e[1])] = frozenset()
    elif kind == "req":
        _, a, ro, named = e
        obj_key, ps = ACTIONS[a]
        if ro is not None:
            add(obj_key, ro)
        for key, u in zip(ps.values(), named):
            if u is not None:
                add(key, u)
        j = list(ACTIONS).index(a)
        rho[j] = _merge(rho[j], ro, named)
    elif kind == "withdraw":
        rho[list(ACTIONS).index(e[1])] = "absent"
    elif kind == "exec":
        op = e[1:]
        if not permitted(regime, s, op):
            return None  # not an admissible event here
        rho[list(ACTIONS).index(op[0])] = "absent"
    return State(tuple(keys), tuple(rho))


def closure(regime, seeds):
    ev = events(regime)
    seen, order, frontier = set(seeds), list(seeds), list(seeds)
    while frontier:
        nxt = []
        for st in frontier:
            for e in ev:
                t = step(regime, st, e)
                if t is not None and t not in seen:
                    seen.add(t)
                    order.append(t)
                    nxt.append(t)
        frontier = nxt
    return order


def reachable(regime):
    return closure(regime, [initial(regime)])


def with_marks(states):
    """Memory states for the revocable regime: each key of each reachable state replaced by any
    subset of its values plus MARK, or kept. Closed under events by construction of `step`."""
    out = set()
    for st in states:
        choices = []
        for k in st.keys:
            real = sorted(k)
            opts = {k}
            for r in range(len(real) + 1):
                for sub in itertools.combinations(real, r):
                    opts.add(frozenset(sub) | {MARK})
            choices.append(sorted(opts, key=lambda x: (len(x), sorted(map(str, x)))))
        for keys in itertools.product(*choices):
            out.add(State(tuple(keys), st.rho))
    return closure("revocable", sorted(out, key=repr))


# --- checks -----------------------------------------------------------------------------------


def check_raw_vs_canonical(regime, exhaustive_len=3, samples=20000, sample_len=9, seed=0):
    ev = events(regime)
    non_exec = [e for e in ev if e[0] != "exec"]
    checked = 0

    def run(history):
        nonlocal checked
        s = initial(regime)
        prefix = []
        for e in history:
            if e[0] == "exec" and not raw_permitted(prefix, e[1:]):
                return  # not admissible in the literal semantics either
            t = step(regime, s, e)
            assert t is not None, ("canonical refused an admissible execution", prefix, e)
            prefix.append(e)
            s = t
            for op in OPS:
                assert permitted(regime, s, op) == raw_permitted(prefix, op), (regime, prefix, op)
            checked += 1

    for n in range(exhaustive_len + 1):
        for history in itertools.product(non_exec, repeat=n):
            run(list(history))
    rng = random.Random(seed)
    for _ in range(samples):
        history, s = [], initial(regime)
        for _ in range(sample_len):
            if rng.random() < 0.25:
                perms = sorted(perm_set(regime, s), key=repr)
                if perms:
                    e = ("exec",) + rng.choice(perms)
                else:
                    e = rng.choice(non_exec)
            else:
                e = rng.choice(non_exec)
            history.append(e)
            s = step(regime, s, e)
        run(history)
    return checked


def moore_classes(regime, states):
    ev = events(regime)
    index = {s: i for i, s in enumerate(states)}
    trans = np.array([[index.get(step(regime, s, e), -1) for e in ev] for s in states])
    perms = [perm_set(regime, s) for s in states]
    ids = {}
    cls = np.array([ids.setdefault(p, len(ids)) for p in perms])
    while True:
        sig = {}
        new = np.empty_like(cls)
        for i in range(len(states)):
            key = (cls[i], tuple(cls[t] if t >= 0 else -1 for t in trans[i]))
            new[i] = sig.setdefault(key, len(sig))
        if len(sig) == len(set(cls.tolist())):
            return new, trans
        cls = new


def safe_relation(regime, mem_states, true_states):
    """R[i, j]: a system remembering mem_states[i] while the truth is true_states[j] never
    executes anything the truth forbids, in any future. Greatest fixed point."""
    ev = events(regime)
    mi = {st: i for i, st in enumerate(mem_states)}
    ti = {st: i for i, st in enumerate(true_states)}
    pm = [perm_set(regime, st) for st in mem_states]
    pt = [perm_set(regime, st) for st in true_states]
    R = np.array([[pm[i] <= pt[j] for j in range(len(true_states))] for i in range(len(mem_states))])
    TM = np.array([[mi[t] if (t := step(regime, st, e)) is not None else -1 for st in mem_states] for e in ev])
    TT = np.array([[ti[t] if (t := step(regime, st, e)) is not None else -1 for st in true_states] for e in ev])
    rows_m, rows_t = np.arange(len(mem_states)), np.arange(len(true_states))
    while True:
        before = int(R.sum())
        for ei, e in enumerate(ev):
            if e[0] == "exec":
                can = TM[ei] >= 0
                follow = R[np.ix_(np.where(can, TM[ei], rows_m), np.where(TT[ei] >= 0, TT[ei], rows_t))]
                R &= (~can)[:, None] | follow
            else:
                R &= R[np.ix_(TM[ei], TT[ei])]
        if int(R.sum()) == before:
            return R


# --- what the theorems predict ----------------------------------------------------------------


def _bare(r):
    return r not in ("absent", "blocked") and r[0] is None and all(x is None for x in r[1])


def predicted_class_static(st: State):
    """Theorem (exact minimal memory), static regime: the reduced canonical state."""
    rho = {a: st.req(a) for a in ACTIONS}
    for a in ACTIONS:
        if _bare(rho[a]) and any(st.key(k) == TOP for k in reads(a)):
            rho[a] = "absent"
    keys = []
    for k in KEYS:
        readers = [a for a in ACTIONS if k in reads(a)]
        keys.append("erased" if all(rho[a] == "blocked" for a in readers) else st.key(k))
    return (tuple(keys), tuple(rho[a] for a in ACTIONS))


def predicted_class_revocable(st: State):
    """Theorem (exact minimal memory), revocable regime: the canonical state itself."""
    return (st.keys, st.rho)


def predicted_safe_static(mem: State, true: State):
    for a in ACTIONS:
        rm, rt = mem.req(a), true.req(a)
        keys_ok = all(mem.key(k) in (true.key(k), TOP) for k in reads(a))
        if rm == "blocked":
            continue
        if rm == rt and keys_ok:
            continue
        if _bare(rt) and rm == "absent" and keys_ok:
            continue
        return False
    return True


def predicted_safe_revocable(mem: State, true: State):
    for k in KEYS:
        if not (mem.key(k) == true.key(k) or MARK in mem.key(k)):
            return False
    for a in ACTIONS:
        rm, rt = mem.req(a), true.req(a)
        if not (rm == rt or rm == "blocked" or (_bare(rt) and rm == "absent")):
            return False
    return True


if __name__ == "__main__":
    print(f"N = {N}")
