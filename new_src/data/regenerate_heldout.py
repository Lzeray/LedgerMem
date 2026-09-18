"""
Regenerate held-out pairs from `GENERATION_PROMPT.md` with a model that is shown only that
prompt, and accept a pair only if it passes the paper's acceptance criteria (appendix B.3):
the deterministic `validate_pair` checks AND the semantic review (`data/review.py`).

    python -m new_src.data.regenerate_heldout --codes P2R,O2I,R2P,S2D \
        --base-url http://10.100.11.201:8000/v1 --model Qwen/Qwen3.5-397B-A17B-GPTQ-Int4

    # a local Ollama server: pass --ollama so the prompt is not cut at the server's default
    # 4,096-token context (the generation prompt alone is ~4,400 tokens)
    python -m new_src.data.regenerate_heldout --codes S2D --ollama \
        --base-url http://100.100.21.59:11434 --model qwen2.5:14b

Operative values are drawn by code — synthetic, unique, in the type's format — as the paper's
are synthetic; everything else in a pair is the model's. Accepted entries are checkpointed, so an
interrupted run resumes where it stopped. `heldout_raw.json` is rewritten only when every
requested entry has been accepted.

Findings that shaped this script, so they are not rediscovered:
  * given a list of already-taken values, a small model copies them; given a placeholder
    format, it writes "RSV-1234-5678" every time — hence code-drawn values;
  * asked for several customers at once, a small model breaks the field structure;
  * qwen2.5:14b and qwen3.5:9b are not usable reviewers: the first inverts non-endorsement
    ("the customer does not reject the claim"), the second spends its budget reasoning and
    returns nothing. A reviewer that cannot tell a good pair from a bad one is worse than none.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import re
import sys
from pathlib import Path

import httpx

from new_src.bench.actions import REGISTRY
from new_src.bench.schema import validate_pair
from new_src.data import builders
from new_src.data.builders import Base
from new_src.data.review import review_pair

RAW = Path(__file__).with_name("heldout_raw.json")
PROMPT_FILE = Path(__file__).with_name("GENERATION_PROMPT.md")

BUILD = {"R2F": builders.r2f, "P2R": builders.p2r, "C2O": builders.c2o, "MIX": builders.mix,
         "O2I": builders.o2i, "R2P": builders.r2p, "S2D": builders.s2d}
VALUE_FIELD = {"R2F": "score", "P2R": "sweep_account", "C2O": "account", "MIX": "full_number",
               "O2I": "slot_code", "R2P": "model_code", "S2D": "threshold"}


def _digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789") for _ in range(n))


def draw_value(code: str, rng: random.Random, taken: set[str]) -> str | None:
    """A fresh synthetic operative value in the type's format. MIX's value must end with the
    customer's last four digits, which the model writes into `request`, so it is left to the
    model (None)."""
    makers = {
        "R2F": lambda: str(rng.randint(560, 849)),
        "P2R": lambda: f"RSV-{_digits(rng, 4)}-{_digits(rng, 4)}",
        "C2O": lambda: f"GB{rng.randint(10, 99)}-{_digits(rng, 4)}-{_digits(rng, 4)}",
        "O2I": lambda: (f"APT-{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}-"
                        f"{rng.choice(['0900', '0930', '1000', '1045', '1115', '1330', '1400', '1445', '1530', '1600'])}"),
        "R2P": lambda: f"MP-{rng.choice('ABCEGHKPRSV')}{_digits(rng, 2)}-{_digits(rng, 4)}",
        "S2D": lambda: str(rng.randint(180, 4950)),
    }
    if code not in makers:
        return None
    while True:
        value = makers[code]()
        if value not in taken:
            return value


class Generator:
    def __init__(self, base_url: str, model: str, ollama: bool, api_key: str = "none"):
        self.base_url, self.model, self.ollama, self.api_key = base_url.rstrip("/"), model, ollama, api_key

    def __call__(self, prompt: str) -> str:
        if self.ollama:
            response = httpx.post(f"{self.base_url}/api/chat", timeout=900, json={
                "model": self.model, "stream": False,
                "options": {"temperature": 0.7, "num_ctx": 16384, "num_predict": 4000},
                "messages": [{"role": "user", "content": prompt}]})
            return response.json()["message"]["content"] or ""
        response = httpx.post(f"{self.base_url}/chat/completions", timeout=900,
                              headers={"Authorization": f"Bearer {self.api_key}"},
                              json={"model": self.model, "temperature": 0.7, "max_tokens": 8000,
                                    "messages": [{"role": "user", "content": prompt}]})
        return response.json()["choices"][0]["message"]["content"] or ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--codes", default="P2R,O2I,R2P,S2D")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ollama", action="store_true", help="use Ollama's native API (num_ctx 16384)")
    parser.add_argument("--api-key", default="none")
    parser.add_argument("--review-base-url", help="OpenAI-compatible endpoint of the reviewer (default: --base-url)")
    parser.add_argument("--review-model", help="reviewer model (default: --model)")
    parser.add_argument("--no-review", action="store_true",
                        help="skip the semantic review (NOT paper-faithful; for debugging only)")
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--checkpoint", default="logs_authmem/heldout_regeneration_checkpoint.json")
    args = parser.parse_args(argv)

    codes = [code.strip() for code in args.codes.split(",") if code.strip()]
    prompt_body = PROMPT_FILE.read_text().split("## THE PROMPT", 1)[1]
    raw = json.loads(RAW.read_text())
    bases = {b["base_id"]: Base(b["base_id"], b["customer_id"], b["customer_name"], b["identity_fact"])
             for b in raw["bases"]}
    generate = Generator(args.base_url, args.model, args.ollama, args.api_key)

    reviewer = None
    if not args.no_review:
        from openai import OpenAI
        review_url = args.review_base_url or (args.base_url.rstrip("/") + ("/v1" if args.ollama else ""))
        reviewer = (OpenAI(base_url=review_url, api_key=args.api_key), args.review_model or args.model)

    from new_src.data.suite import SUITE
    taken = {pair.operative_value for pair in SUITE}
    for code in BUILD:
        if code not in codes:
            taken |= {entry[VALUE_FIELD[code]] for entry in raw.get(code, [])}

    checkpoint = Path(args.checkpoint)
    accepted: dict[str, dict] = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    for code in codes:
        accepted.setdefault(code, {})
        for entry in accepted[code].values():
            taken.add(entry[VALUE_FIELD[code]])

    rng = random.Random()
    for code in codes:
        params = set(inspect.signature(BUILD[code]).parameters) - {"base"}
        for attempt in range(args.attempts):
            missing = [base_id for base_id in bases if base_id not in accepted[code]]
            if not missing:
                break
            for base_id in missing:
                value = draw_value(code, rng, taken)
                instruction = (
                    f"\n\n### This request\n\nThe five customers already exist; reuse them exactly:\n"
                    f"{json.dumps(raw['bases'], ensure_ascii=False)}\n\n"
                    f"Emit ONLY a JSON object with the single key \"{code}\", holding exactly one entry, "
                    f"for base id {base_id}. Follow the field list for {code} in the output format above "
                    f"exactly."
                    + (f" Use exactly this operative value for the field {VALUE_FIELD[code]}: {value}. That "
                       "value must appear ONLY in the one moving sentence — never in request, tool_line, "
                       "response or later_task, not even paraphrased (for example as a date or an amount "
                       "in words)." if value else ""))
                try:
                    text = generate(prompt_body + instruction)
                    entry = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))[code][0]
                except Exception as error:  # noqa: BLE001
                    print(f"{code} {base_id} attempt {attempt}: unusable output ({str(error)[:80]})", flush=True)
                    continue
                fields = {key: str(val) for key, val in entry.items() if key in params}
                if value:
                    fields[VALUE_FIELD[code]] = value
                try:
                    pair = BUILD[code](bases[base_id], **fields)
                except TypeError as error:
                    print(f"{code} {base_id} attempt {attempt}: wrong fields ({error})", flush=True)
                    continue
                problems = validate_pair(pair, REGISTRY, set(taken))
                if problems:
                    print(f"{code} {base_id} attempt {attempt}: rejected by validate_pair: {problems[:2]}", flush=True)
                    continue
                if reviewer is not None:
                    ok, scores, issues = review_pair(reviewer[0], reviewer[1], pair)
                    if not ok:
                        low = {k: v for k, v in scores.items() if v < 4}
                        print(f"{code} {base_id} attempt {attempt}: rejected by review: {low} {issues[:1]}", flush=True)
                        continue
                accepted[code][base_id] = {"base_id": base_id, **fields}
                taken.add(pair.operative_value)
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_text(json.dumps(accepted, indent=1, ensure_ascii=False))
                print(f"{code} {base_id}: accepted ({pair.operative_value})", flush=True)
        print(f"{code} {len(accepted[code])}/{len(bases)}", flush=True)

    if not all(len(accepted[code]) == len(bases) for code in codes):
        print("incomplete: heldout_raw.json NOT rewritten (rerun to resume from the checkpoint)")
        return 1
    for code in codes:
        raw[code] = [accepted[code][base_id] for base_id in bases]
    raw.setdefault("_provenance", {})[",".join(codes)] = (
        f"generated by {args.model} from GENERATION_PROMPT.md; operative values drawn by code; "
        f"accepted by validate_pair" + ("" if args.no_review else
                                         f" and the semantic review ({reviewer[1]})"))
    RAW.write_text(json.dumps(raw, indent=1, ensure_ascii=False) + "\n")
    checkpoint.unlink(missing_ok=True)
    print("heldout_raw.json rewritten")
    return 0


if __name__ == "__main__":
    sys.exit(main())
