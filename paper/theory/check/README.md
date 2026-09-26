# Brute-force checks of the theory

Each theorem in `paper/theory/` that can be stated on a finite instance is checked here by
enumeration. Nothing is sampled except where noted.

```bash
cd paper/theory/check
../../../.venv/bin/python run_checks.py 3 --safe      # semantics, exact memory, safe forgetting (N = 3)
../../../.venv/bin/python run_checks.py 4             # semantics and exact memory (N = 4)
../../../.venv/bin/python gate_model.py 3 3           # the gate's steps 1-2 vs the semantics
```

- `authority.py` — the permission semantics twice: literally from a history, and as the canonical
  state machine the theorems describe; Moore minimisation for Nerode classes; the greatest
  safe-replacement relation; what each theorem predicts.
- `run_checks.py` — compares them and prints PASS/FAIL per theorem.
- `gate_model.py` — the gate of `new_src/bench/gate.py` (steps 1-2, no customer) on records written
  faithfully with any pattern of lowered customer messages, against the semantics; and the same
  gate without the veto on lowered customer words, which has a counterexample.
