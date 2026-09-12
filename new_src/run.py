"""
Command line for the AuthMem-Bench banking re-implementation.

    python -m new_src.run validate
        Deterministic schema validation of all 35 pairs: the carrier swap, the operative
        value's novelty and global uniqueness, the frozen-role gold labels, closed-world
        tool schemas, and that the later action actually consumes the contested value.
        Runs entirely offline — no model, no database.

    python -m new_src.run b --condition baseline
    python -m new_src.run b --condition gate
        Module B, action time. `baseline` is the unprotected condition (washed memory, no
        authority metadata, tools in the agent's hands); `gate` routes every action through
        the authority gate. Compare their ASR/TSR.

    python -m new_src.run a
        Module A, write time: consolidate each history and score whether the focal claim was
        upgraded, preserved or dropped.

    python -m new_src.run c --condition gate-predicted
        Module C, end to end: consolidation, automatic labeling, retrieval, action.

    python -m new_src.run check
        Validity check on the live stack: with the defense off the H- attack must actually
        succeed, and with the defense on the H+ task must succeed without a confirmation
        round-trip. A pair that fails either half is a bad test, not evidence of a good
        defense — this is the check that catches a suite which cannot fail.

Filters (--categories, --bases, --variants, --limit) apply to every subcommand, so a smoke
run over two categories is one flag away from the full suite.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import pathlib
import sys

from new_src.bench import module_a, module_b, module_c
from new_src.bench.actions import REGISTRY
from new_src.bench.engine import make_client
from new_src.bench.logging_utils import run_dir, transcript, write_summary
from new_src.bench.metrics import action_summary, append_jsonl, by_category, load_jsonl
from new_src.bench.metrics import write_summary as write_metrics
from new_src.bench.module_b import Condition
from new_src.bench.schema import validate_pair
from new_src.config import ACTION_MODEL, BASE_URL
from new_src.data.license_attacks import LICENSE_ATTACKS
from new_src.data.suite import SUITE

CONDITIONS: dict[str, Condition] = {
    "baseline": module_b.BASELINE,
    "baseline-attributed": module_b.BASELINE_ATTRIBUTED,
    "gold-prompted": module_b.GOLD_PROMPTED,
    "heuristic-prompted": module_b.HEURISTIC_PROMPTED,
    "heuristic-washed": module_b.HEURISTIC_WASHED,
    "gate": module_b.GATE_GOLD,
    "gate-washed": module_b.GATE_GOLD_WASHED,
    "gate-predicted": module_b.GATE_PREDICTED,
    "gate-license": module_b.GATE_LICENSE,
    "gate-license-declared": module_b.GATE_LICENSE_DECLARED,
    "gate-license-model": module_b.GATE_LICENSE_MODEL,
    "gate-native": module_b.GATE_NATIVE,
    "gate-native-predicted": module_b.GATE_NATIVE_PREDICTED,
    "gate-heuristic": module_b.GATE_HEURISTIC,
    "memory-off": module_b.MEMORY_OFF,
}


def _condition_dir(args, condition) -> str:
    """Keep the two suites' records in separate directories. They answer different questions and
    averaging them together produces a number that means nothing."""
    return condition.name + ("" if suite_for(args) is SUITE else "__licence")


def suite_for(args) -> list:
    """`--suite core` is the 35-pair taxonomy suite; `licence` is the 15 pairs that attack the
    gate's licence check itself (see data/license_attacks.py); `all` is both."""
    choice = getattr(args, "suite", "core")
    if choice == "licence":
        return LICENSE_ATTACKS
    if choice == "all":
        return [*SUITE, *LICENSE_ATTACKS]
    return SUITE


def select_pairs(args) -> list:
    pairs = suite_for(args)
    if args.categories:
        wanted = {code.strip().upper() for code in args.categories.split(",")}
        pairs = [pair for pair in pairs if pair.category.code in wanted]
    if args.bases:
        wanted = {base.strip().upper() for base in args.bases.split(",")}
        pairs = [pair for pair in pairs if pair.base_id in wanted]
    if getattr(args, "families", ""):
        wanted = {f.strip().upper() for f in args.families.split(",")}
        pairs = [pair for pair in pairs if pair.pair_id[0].upper() in wanted]
    if args.limit:
        pairs = pairs[: args.limit]
    return pairs


def variants(args) -> list[str]:
    return [value.strip() for value in args.variants.split(",")]


def _validate_license_pair(pair) -> list[str]:
    """The licence-attack pairs are built from one focal sentence that moves between carriers,
    so the checks that apply are: the sentence occurs in exactly one message per variant,
    removing it leaves identical episodes, and the action really acts on the object the pair
    contests."""
    problems = []
    minus, plus = pair.episode("H-"), pair.episode("H+")
    for variant, episode in (("H-", minus), ("H+", plus)):
        carrying = [m for m in episode.messages if pair.focal_quote in m.content]
        if len(carrying) != 1:
            problems.append(f"{variant}: focal sentence appears in {len(carrying)} messages, expected 1")
    stripped = [
        [(m.role, m.content.replace(pair.focal_quote, "").strip()) for m in episode.messages]
        for episode in (minus, plus)
    ]
    if stripped[0] != stripped[1]:
        problems.append("carrier-swap violated: episodes differ after removing the focal sentence")
    if pair.object_ref and pair.object_ref not in {str(v) for v in pair.target_arguments.values()}:
        problems.append("the action does not act on the object the pair contests")
    if pair.target_tool not in REGISTRY:
        problems.append(f"unknown tool {pair.target_tool!r}")
    return problems


@contextlib.contextmanager
def exclusive_run():
    """Refuse to start while another run is already going.

    Every episode begins by emptying both memory tables — that is what makes an episode
    independent of the one before it. Two runs sharing the database therefore delete each
    other's records mid-episode, and the gate, which reads its verdict from those tables, starts
    answering from whatever happened to survive.

    It is not a hypothetical. A night's programme was launched twice by accident and the two
    processes ran interleaved for three and a half hours. The damage is legible in the results:
    conditions that never read the database disagreed between the two copies on 0 of 118
    repeated episodes, while gate conditions disagreed on 29 of 139. The unprotected numbers
    survived; every number for the defense had to be thrown away.

    flock is released by the operating system when the process dies, so a crash cannot leave a
    stale lock behind.
    """
    import fcntl

    path = pathlib.Path(".authmem.lock")
    handle = open(path, "w")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("  Another run is already using the memory database.")
            print("  Runs cannot overlap: each episode wipes the memory tables, so a second run")
            print("  would delete this one's records mid-episode and corrupt both sets of")
            print("  results. Wait for it to finish, or stop it, then start this one.")
            raise SystemExit(2) from None
        handle.write(str(os.getpid()))
        handle.flush()
        yield
    finally:
        handle.close()


class EpisodeFailures:
    """Keeps one failing episode from costing a whole run.

    A single episode can die for reasons that have nothing to do with the rest of the sweep: a
    model returning something unparsable, a request timing out, a column type that does not yet
    know about a value being written. Before this, any of those ended the phase and took every
    remaining episode with it — a claim type missing from one enum cost an entire Module C run
    partway through.

    A failed episode is deliberately NOT recorded. `--resume` skips what is in the log, so
    writing a failure would mean never retrying it; leaving it out means the next run picks it
    up. The cost is that a permanently broken episode is retried on every run, which is why the
    breaker below exists.
    """

    def __init__(self, consecutive_limit: int = 5):
        self.failures: list[tuple[str, str]] = []
        self.consecutive = 0
        self.limit = consecutive_limit

    def record(self, name: str, error: BaseException) -> bool:
        """Note a failure. Returns False when the phase should stop trying."""
        detail = f"{type(error).__name__}: {error}"
        self.failures.append((name, detail))
        self.consecutive += 1
        print(f"\n  !! {name} failed and was not recorded: {detail[:200]}")
        if self.consecutive >= self.limit:
            print(f"  !! {self.consecutive} episodes failed in a row — stopping this phase.")
            print("     Consecutive failures mean something systematic, not bad luck, and")
            print("     grinding through the remaining episodes would waste the night on it.")
            return False
        print("     Continuing with the next episode; re-run with --resume to retry this one.")
        return True

    def succeeded(self) -> None:
        self.consecutive = 0

    def report(self) -> None:
        if not self.failures:
            return
        print(f"\n  {len(self.failures)} episode(s) failed and were NOT recorded:")
        for name, detail in self.failures:
            print(f"    - {name}: {detail[:160]}")
        print("  Re-run the same command with --resume to retry exactly these.")


def cmd_validate(args) -> int:
    seen: set[str] = set()
    failures = 0
    for pair in select_pairs(args):
        if not hasattr(pair, "operative_value"):
            problems = _validate_license_pair(pair)
            status = "ok" if not problems else "FAILED"
            print(f"  {pair.pair_id:<10} {pair.family[:40]:<42} {status}")
            for problem in problems:
                print(f"      - {problem}")
            failures += bool(problems)
            continue
        problems = validate_pair(pair, REGISTRY, seen)
        status = "ok" if not problems else "FAILED"
        print(f"  {pair.pair_id:<10} {pair.category.transition:<32} {status}")
        for problem in problems:
            print(f"      - {problem}")
        failures += bool(problems)
    total = len(select_pairs(args))
    print(f"\n  {total - failures}/{total} pairs pass deterministic schema validation.")
    return 1 if failures else 0


def cmd_action(args, module: str) -> int:
    condition = CONDITIONS[args.condition]
    if args.confirm:
        condition = Condition(**{**condition.__dict__, "confirm_followup": True})
    if module == "C" and condition.label_source not in ("gold", "predicted", "channel-typed"):
        print("Module C supports label sources gold, predicted and channel-typed "
              "(--condition gate, gate-predicted or gate-license-model).")
        return 2

    client = make_client()
    pairs = select_pairs(args)
    directory = run_dir(args.model, f"module_{module.lower()}", _condition_dir(args, condition))
    records = []

    # --resume: skip episodes already recorded for this exact condition. A long sweep on a
    # local GPU can die halfway (a machine reboot, a driver lock-up) and re-running the whole
    # thing wastes an hour and, worse, doubles some pairs in the records.
    done: set[tuple[str, str]] = set()
    if args.resume:
        for record in load_jsonl(directory / "episodes.jsonl"):
            done.add((record["pair_id"], record["variant"]))
        planned = sum(1 for pair in pairs for variant in variants(args) if (pair.pair_id, variant) not in done)
        print(f"  resume: {len(done)} episodes already recorded, {planned} left to run")
        records.extend(load_jsonl(directory / "episodes.jsonl"))

    # One line per episode even under --quiet. Without it a long run writes nothing to its log
    # between phase banners, and a working sweep is indistinguishable from a hung one — which is
    # exactly how it looked from outside the first time a night run was left going.
    planned = [(pair, variant) for pair in pairs for variant in variants(args)
               if (pair.pair_id, variant) not in done]
    total = len(planned)
    position = 0

    failures = EpisodeFailures()
    stop = False
    for pair in pairs:
        if stop:
            break
        for variant in variants(args):
            if (pair.pair_id, variant) in done:
                continue
            position += 1
            name = f"{pair.pair_id}_{'minus' if variant == 'H-' else 'plus'}"
            runner = module_b.run_episode if module == "B" else module_c.run_episode
            try:
                # verbose is always on: it is what fills the per-episode log file. --quiet only
                # stops the transcript being echoed to the terminal.
                with transcript(args.model, f"module_{module.lower()}", _condition_dir(args, condition), name,
                                echo=not args.quiet):
                    kwargs = {"model": args.model, "verbose": True}
                    if module == "B":
                        kwargs["module"] = "B"
                    record = runner(client, pair, variant, condition, **kwargs)
            except Exception as error:  # noqa: BLE001 - one episode must not end the sweep
                if not failures.record(name, error):
                    stop = True
                    break
                continue
            failures.succeeded()
            append_jsonl(directory / "episodes.jsonl", record)
            records.append(record)
            expected = "required" if variant == "H+" else "PROHIBITED"
            verdict = "performed" if record.performed else "not performed"
            print(f"    [{position}/{total}] {pair.pair_id} {variant}  {verdict}"
                  f"  (this action was {expected})", flush=True)
    failures.report()

    summary = action_summary([_as_record(r) for r in records])
    lines = [
        f"{'='*72}",
        f"  Module {module} — condition '{args.condition}' ({condition.name})",
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
    path = write_summary(args.model, f"module_{module.lower()}", _condition_dir(args, condition), lines)
    print("\n" + "\n".join(lines))
    print(f"\n  summary written to {path}")
    return 0


def cmd_write(args) -> int:
    client = make_client()
    pairs = select_pairs(args)
    directory = run_dir(args.model, "module_a", "consolidation")
    records = []

    failures = EpisodeFailures()
    stop = False
    for pair in pairs:
        if stop:
            break
        for variant in variants(args):
            name = f"{pair.pair_id}_{'minus' if variant == 'H-' else 'plus'}"
            try:
                with transcript(args.model, "module_a", "consolidation", name, echo=not args.quiet):
                    record = module_a.run_episode(client, pair, variant, verbose=True)
            except Exception as error:  # noqa: BLE001
                if not failures.record(name, error):
                    stop = True
                    break
                continue
            failures.succeeded()
            append_jsonl(directory / "write_records.jsonl", record)
            records.append(record)
    failures.report()

    summary = write_metrics(records)
    lines = [
        f"{'='*72}",
        "  Module A — write-time consolidation",
        f"  endpoint: {BASE_URL}   model: {args.model}",
        f"  {len(records)} episodes, records appended to {directory / 'write_records.jsonl'}",
        "",
        f"  overall   Upgrade-all={_pct(summary['Upgrade_all'])}  Ret-={_pct(summary['Ret_minus'])}"
        f"  Ret+={_pct(summary['Ret_plus'])}  FAU={_pct(summary['FAU'])}",
    ]
    for category, values in by_category(records, write_metrics).items():
        lines.append(
            f"    {category:<5} Upgrade-all={_pct(values['Upgrade_all'])}  Ret-={_pct(values['Ret_minus'])}"
            f"  Ret+={_pct(values['Ret_plus'])}"
        )
    path = write_summary(args.model, "module_a", "consolidation", lines)
    print("\n" + "\n".join(lines))
    print(f"\n  summary written to {path}")
    return 0


def cmd_check(args) -> int:
    """The validity check: an attack that does not succeed with the defense off proves
    nothing about the defense, and a task the defense blocks on the authorized side is a
    broken test rather than a safe one."""
    client = make_client()
    pairs = select_pairs(args)
    results = []

    for pair in pairs:
        with transcript(args.model, "check", "attack_succeeds_unprotected", f"{pair.pair_id}_minus",
                        echo=not args.quiet):
            unprotected = module_b.run_episode(client, pair, "H-", module_b.BASELINE,
                                               model=args.model, verbose=True)
        with transcript(args.model, "check", "task_succeeds_gated", f"{pair.pair_id}_plus",
                        echo=not args.quiet):
            gated = module_b.run_episode(client, pair, "H+", module_b.GATE_GOLD,
                                         model=args.model, verbose=True)
        results.append((pair, unprotected.performed, gated.performed and not gated.confirmation_requested))

    print(f"\n{'='*72}\n  Validity check ({len(results)} pairs)")
    print(f"  {'pair':<10}{'H- succeeds unprotected':>26}{'H+ succeeds gated':>20}   verdict")
    passed = 0
    for pair, attack_ok, task_ok in results:
        verdict = "valid" if (attack_ok and task_ok) else "REVIEW"
        passed += attack_ok and task_ok
        print(f"  {pair.pair_id:<10}{str(attack_ok):>26}{str(task_ok):>20}   {verdict}")
    print(f"\n  {passed}/{len(results)} pairs valid on this model.")
    return 0


def cmd_null(args) -> int:
    """Null control: the same H- episodes with the contested record removed. Every one of them
    should fail to produce the action. Any that still fires is a pair whose later task alone
    triggers the action, which would inflate ASR for a reason that has nothing to do with
    authority."""
    client = make_client()
    pairs = select_pairs(args)
    fired = []
    null_dir = "baseline_without_contested_record" + ("" if suite_for(args) is SUITE else "__licence")
    directory = run_dir(args.model, "null_control", null_dir)

    # The null control resumes like every other phase. It did not, and a night's programme
    # started twice simply appended a second copy of all 35 episodes — the same pair recorded
    # twice, silently, with no way to tell which reading was which.
    done: set[str] = set()
    if args.resume:
        existing = load_jsonl(directory / "episodes.jsonl")
        done = {row["pair_id"] for row in existing}
        fired = [row["pair_id"] for row in existing if row["performed"]]
        print(f"  resume: {len(done)} pairs already checked")

    failures = EpisodeFailures()
    for pair in pairs:
        if pair.pair_id in done:
            continue
        try:
            with transcript(args.model, "null_control", null_dir,
                            f"{pair.pair_id}_minus", echo=not args.quiet):
                record = module_b.run_episode(client, pair, "H-", module_b.BASELINE,
                                              model=args.model, verbose=True, drop_focal=True)
        except Exception as error:  # noqa: BLE001
            if not failures.record(f"{pair.pair_id}_minus", error):
                break
            continue
        failures.succeeded()
        # Recorded like any other episode, so ASR can later be recomputed over only the pairs
        # whose null control came back clean.
        append_jsonl(directory / "episodes.jsonl", record)
        if record.performed:
            fired.append(pair.pair_id)
        print(f"    {pair.pair_id}  {'FIRED ANYWAY' if record.performed else 'did not fire'}",
              flush=True)
    failures.report()

    lines = [
        f"{'='*72}",
        "  Null control — H- with the contested record removed",
        f"  endpoint: {BASE_URL}   model: {args.model}",
        f"  {len(pairs)} pairs; the action should fire in none of them.",
        "",
        f"  fired anyway: {len(fired)}/{len(pairs)}" + (f"  -> {', '.join(fired)}" if fired else ""),
    ]
    path = write_summary(args.model, "null_control", null_dir, lines)
    print("\n" + "\n".join(lines))
    print(f"\n  summary written to {path}")
    return 0


class _Row(dict):
    """Attribute access over a recorded episode dict, so summaries can treat replayed records
    from the log exactly like ones produced in this process."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def _as_record(record):
    return record if not isinstance(record, dict) else _Row(record)


def _pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="new_src.run", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["validate", "a", "b", "c", "check", "null"])
    parser.add_argument("--condition", default="baseline", choices=sorted(CONDITIONS))
    parser.add_argument("--categories", default="", help="Comma-separated category codes, e.g. R2F,S2D")
    parser.add_argument("--bases", default="", help="Comma-separated base ids, e.g. B1,B2")
    parser.add_argument("--variants", default="H-,H+", help="Which variants to run")
    parser.add_argument("--limit", type=int, default=0, help="Run at most this many pairs")
    parser.add_argument("--model", default=ACTION_MODEL)
    parser.add_argument("--confirm", action="store_true",
                        help="Gate conditions only: run one scripted confirmation round-trip (beyond the paper)")
    parser.add_argument("--suite", default="core", choices=["core", "licence", "all"],
                        help="Which pairs to run: the taxonomy suite, the licence-attack suite, or both")
    parser.add_argument("--families", default="",
                        help="Licence suite only: comma-separated family letters, e.g. A,C")
    parser.add_argument("--resume", action="store_true",
                        help="Skip episodes already recorded for this condition and run only what is missing")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "validate":
        return cmd_validate(args)          # offline: no database, no lock needed
    with exclusive_run():
        if args.command == "null":
            return cmd_null(args)
        if args.command == "a":
            return cmd_write(args)
        if args.command == "b":
            return cmd_action(args, "B")
        if args.command == "c":
            return cmd_action(args, "C")
        return cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
