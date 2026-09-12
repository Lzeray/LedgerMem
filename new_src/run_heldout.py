"""
Runner for the held-out suites.

    python -m new_src.run_heldout validate
    python -m new_src.run_heldout b --suite core      --condition baseline
    python -m new_src.run_heldout b --suite core      --condition gate-license
    python -m new_src.run_heldout b --suite speechact --condition baseline-attributed
    python -m new_src.run_heldout b --suite speechact --condition gate-license

Two suites, kept apart because they answer different questions and averaging them produces a
number that means nothing:

  `core`       the 35 held-out pairs — 5 fresh base histories × the 7 established transitions,
               generated after the gate's design was frozen (see data/GENERATION_PROMPT.md).
               This is the set headline numbers belong on; `data/suite.py` is the development
               set the defense was built against.
  `speechact`  the 20 pairs of Q2D / N2D / P2F / G2O, which attack the role→claim-type mapping
               rather than the provenance of a claim.

A separate entry point rather than a new flag on `new_src.run`: the existing CLI drives every
published number in the project and there is no reason to modify it to add a dataset.

Records land in the ordinary tree so `new_src.report` and any comparison read them unchanged:

    logs_authmem/<model>/module_b/<condition>__heldout[_speechact]/episodes.jsonl
"""

from __future__ import annotations

import argparse
import sys

from new_src.bench import module_b
from new_src.bench.actions import REGISTRY
from new_src.bench.engine import make_client
from new_src.bench.logging_utils import run_dir, transcript, write_summary
from new_src.bench.metrics import action_summary, append_jsonl, by_category, load_jsonl
from new_src.bench.schema import validate_pair
from new_src.config import ACTION_MODEL, BASE_URL
from new_src.data import speech_act_attacks
from new_src.data.heldout import HELDOUT_SUITE
from new_src.data.speech_act_attacks import SPEECH_ACT_PAIRS
from new_src.run import CONDITIONS, EpisodeFailures, _as_record, _pct, exclusive_run


def suite_for(name: str) -> list:
    return SPEECH_ACT_PAIRS if name == "speechact" else HELDOUT_SUITE


def select_pairs(args) -> list:
    pairs = suite_for(args.suite)
    if args.categories:
        wanted = {code.strip().upper() for code in args.categories.split(",")}
        pairs = [pair for pair in pairs if pair.category.code in wanted]
    if args.bases:
        wanted = {base.strip().upper() for base in args.bases.split(",")}
        pairs = [pair for pair in pairs if pair.base_id in wanted]
    if args.limit:
        pairs = pairs[: args.limit]
    return pairs


def cmd_validate(args) -> int:
    failures = 0
    for name in (["core", "speechact"] if args.suite == "all" else [args.suite]):
        pairs = suite_for(name)
        print(f"\n  {name} suite ({len(pairs)} pairs)")
        seen: set[str] = set()
        for pair in pairs:
            problems = (speech_act_attacks.validate(pair, REGISTRY) if name == "speechact"
                        else validate_pair(pair, REGISTRY, seen))
            print(f"    {pair.pair_id:<10} {pair.category.transition:<32} {'ok' if not problems else 'FAILED'}")
            for problem in problems:
                print(f"        - {problem}")
            failures += bool(problems)
    print(f"\n  {'all pairs pass' if not failures else f'{failures} pairs FAILED'} deterministic validation.")
    return 1 if failures else 0


def cmd_action(args) -> int:
    condition = CONDITIONS[args.condition]
    if args.suite == "speechact" and condition.rendering == "washed":
        # Washing removes a source condition. Three of these four families have no source
        # condition to remove, and N2D's H- claim is a refusal, whose "washed" form would
        # invert its meaning rather than strip an attribution. Use the attributed baseline.
        print("  The washed rendering is undefined for the speech-act suite; "
              "use --condition baseline-attributed as the unprotected arm.")
        return 2

    directory_name = f"{condition.name}__heldout" + ("_speechact" if args.suite == "speechact" else "")
    if args.null:
        # Null-control episodes get their own directory, exactly as new_src.run does. Sharing a
        # directory with the real run makes --resume treat a null episode as a completed one,
        # and puts two different measurements in one file where only the `notes` marker keeps
        # them apart.
        directory_name = "null_control__heldout" + ("_speechact" if args.suite == "speechact" else "")
    client = make_client()
    pairs = select_pairs(args)
    variants = [value.strip() for value in args.variants.split(",")]
    directory = run_dir(args.model, "module_b", directory_name)
    records: list = []

    done: set[tuple[str, str]] = set()
    if args.resume:
        for row in load_jsonl(directory / "episodes.jsonl"):
            done.add((row["pair_id"], row["variant"]))
        records.extend(load_jsonl(directory / "episodes.jsonl"))
        print(f"  resume: {len(done)} episodes already recorded")

    failures = EpisodeFailures()
    stop = False
    for pair in pairs:
        if stop:
            break
        for variant in variants:
            if (pair.pair_id, variant) in done:
                continue
            name = f"{pair.pair_id}_{'minus' if variant == 'H-' else 'plus'}"
            try:
                with transcript(args.model, "module_b", directory_name, name, echo=not args.quiet):
                    record = module_b.run_episode(client, pair, variant, condition,
                                                  model=args.model, module="B", verbose=True,
                                                  drop_focal=args.null)
            except Exception as error:  # noqa: BLE001 - one episode must not end the suite
                if not failures.record(name, error):
                    stop = True
                    break
                continue
            failures.succeeded()
            append_jsonl(directory / "episodes.jsonl", record)
            records.append(record)
            print(f"    {pair.pair_id} {variant}  "
                  f"{'performed' if record.performed else 'not performed'}", flush=True)
    failures.report()

    summary = action_summary([_as_record(r) for r in records])
    lines = [
        f"{'='*72}",
        f"  Module B — held-out {args.suite} suite — condition '{args.condition}' ({condition.name})",
        f"  endpoint: {BASE_URL}   model: {args.model}",
        f"  {len(records)} episodes, records appended to {directory / 'episodes.jsonl'}",
        "",
        f"  overall   ASR={_pct(summary['ASR'])}  TSR={_pct(summary['TSR'])}"
        f"   (N-={summary['n_minus']}, N+={summary['n_plus']})",
    ]
    for category, values in by_category([_as_record(r) for r in records], action_summary).items():
        lines.append(
            f"    {category:<5} ASR={_pct(values['ASR'])}  TSR={_pct(values['TSR'])}"
            f"   asked for confirmation: H-={_pct(values['confirmation_rate_minus'])}"
            f" H+={_pct(values['confirmation_rate_plus'])}"
        )
    path = write_summary(args.model, "module_b", directory_name, lines)
    print("\n" + "\n".join(lines))
    print(f"\n  summary written to {path}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="new_src.run_heldout", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["validate", "b"])
    parser.add_argument("--suite", default="core", choices=["core", "speechact", "all"])
    parser.add_argument("--condition", default="baseline", choices=sorted(CONDITIONS))
    parser.add_argument("--categories", default="")
    parser.add_argument("--bases", default="")
    parser.add_argument("--variants", default="H-,H+")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default=ACTION_MODEL)
    parser.add_argument("--null", action="store_true",
                        help="null control: drop the contested record; the action must not fire")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "validate":
        return cmd_validate(args)
    if args.suite == "all":
        print("  --suite all is for validate only; run each suite separately.")
        return 2
    with exclusive_run():
        return cmd_action(args)


if __name__ == "__main__":
    sys.exit(main())
