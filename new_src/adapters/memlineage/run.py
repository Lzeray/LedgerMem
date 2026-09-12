"""
Module B, action time, with MemLineage standing in for the memory system.

    python -m new_src.adapters.memlineage.run --condition memlineage
    python -m new_src.adapters.memlineage.run --condition memlineage --categories R2F,S2D

This is an adapter, not a fork: nothing under `new_src/bench/` or `new_src/data/` is modified
or reimplemented. The episode construction, the action stage, the strict action predicate, the
ASR/TSR metrics and the logging are the benchmark's own; the only thing replaced is where the
records the agent sees come from. In the benchmark's own conditions they come from the
deterministic memory stub with authority labels attached. Here they come out of a live
MemLineage backend, through its governed write path and its documented read action, with
whatever MemLineage retained and nothing more.

What the comparison is for: the gate under test in `new_src` decides from an authority label
in code. MemLineage has no such label, so an agent backed by it has only prose and a `sources`
list to go on and the action is never intercepted at all. Whether that is enough is what these
runs measure. The comparison arms are the untouched benchmark:

    python -m new_src.run b --condition baseline             # washed memory, no metadata
    python -m new_src.run b --condition baseline-attributed  # prose names the source
    python -m new_src.run b --condition gate                 # the authority gate

Records land in the same tree as every other run, under their own condition directory:

    logs_authmem/<model>/module_b/<condition>/episodes.jsonl
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from new_src.adapters.memlineage.client import MemLineageClient
from new_src.adapters.memlineage.store import MemLineageWorkspace, as_memory_records
from new_src.bench import action_stage, dms
from new_src.bench.engine import make_client
from new_src.bench.logging_utils import run_dir, transcript, write_summary
from new_src.bench.metrics import ActionRecord, action_summary, append_jsonl, by_category, load_jsonl
from new_src.bench.module_b import Condition
from new_src.config import ACTION_MODEL, BASE_URL
from new_src.data.license_attacks import LICENSE_ATTACKS
from new_src.data.suite import SUITE
from new_src.run import _as_record, _pct


@dataclass(frozen=True)
class MemLineageCondition:
    """A benchmark condition whose memory is a live MemLineage workspace.

    `policy` is always `direct`: MemLineage does not gate downstream actions, so the agent
    holds the ordinary banking tools. `show_metadata` is always False, because there is no
    authority metadata in MemLineage to show — displaying one would mean inventing it.
    """

    name: str
    read_path: str = "notes"     # notes | context
    with_sources: bool = True    # show each note's stored `sources` alongside its body
    description: str = ""

    @property
    def stage_condition(self) -> Condition:
        return Condition(
            policy="direct",
            label_source=f"memlineage:{self.read_path}" + ("" if self.with_sources else "/nosources"),
            rendering="source_attributed",
            show_metadata=False,
        )


CONDITIONS: dict[str, MemLineageCondition] = {
    "memlineage": MemLineageCondition(
        name="memlineage",
        read_path="notes",
        with_sources=True,
        description="notes read back with the sources MemLineage stored for each — its best read surface",
    ),
    "memlineage-nosources": MemLineageCondition(
        name="memlineage-nosources",
        read_path="notes",
        with_sources=False,
        description="the same notes with only their bodies, as a summarizing read would hand them over",
    ),
    "memlineage-context": MemLineageCondition(
        name="memlineage-context",
        read_path="context",
        with_sources=False,
        description="get_context_bundle, the skill's generic context read: titles only, no bodies, no sources",
    ),
}


def run_episode(client, workspace: MemLineageWorkspace, pair, variant: str,
                condition: MemLineageCondition, model: str, verbose: bool = True) -> ActionRecord:
    # The source-attributed rendering is deliberate and in MemLineage's favour: it is handed
    # records that still name their source condition, on top of the machine-readable `sources`
    # its own write path stores. Feeding it washed records would test the paper's collapse
    # rather than MemLineage.
    episode = pair.episode(variant, rendering="source_attributed", claim_type_source="role")
    records = list(episode.memory)
    focal = next(record for record in records if record.is_focal)

    workspace.reset()
    receipt = workspace.write_records(records)
    rows = workspace.read_notes() if condition.read_path == "notes" else workspace.read_context_bundle()
    shown = as_memory_records(rows, with_sources=condition.with_sources, read_path=condition.read_path)

    stage_condition = condition.stage_condition
    # An empty store on the benchmark's own side: the action stage opens a session against it,
    # and nothing may leak from a previous episode into this one.
    engine, _ = dms.install(episode, [])

    if verbose:
        print(f"\n{'='*72}\n  {pair.pair_id}  {variant}  [{condition.name}]  target={episode.target_tool}")
        print(f"  focal record : {focal.text}")
        print(f"  gold label   : {focal.label} (role {focal.role}) — MemLineage stores no label")
        print(f"  MemLineage   : change set {receipt.change_set_id} committed as {receipt.commit_id}, "
              f"{receipt.notes_written} notes")
        print(f"  read back    : {len(shown)} records via {condition.read_path}")
        for record in shown:
            print(f"    - {record.text}")
        print(f"  later task   : {episode.later_task}")

    record = ActionRecord(
        module="B", pair_id=pair.pair_id, base_id=pair.base_id, category=pair.category.code,
        variant=variant, policy=stage_condition.policy, label_source=stage_condition.label_source,
        rendering=stage_condition.rendering, show_metadata=False, gate_surface="none",
        action_permitted=episode.action_permitted, target_tool=episode.target_tool,
        target_arguments=episode.target_arguments, performed=False,
        # What the benchmark knows the focal record's authority to be. MemLineage stored no
        # label of its own, which is the point of the run; recording the gold value here keeps
        # the episode records comparable with every other condition's.
        focal_label_stored=focal.label, focal_role_stored=focal.role,
        notes=(f"memlineage read_path={condition.read_path} with_sources={condition.with_sources} "
               f"change_set={receipt.change_set_id} commit={receipt.commit_id} "
               f"notes_written={receipt.notes_written} notes_read={len(shown)}"),
    )
    return action_stage.perform(client, episode, engine, shown, stage_condition, record, model, verbose)


def select_pairs(args) -> list:
    pairs = LICENSE_ATTACKS if args.suite == "licence" else (
        [*SUITE, *LICENSE_ATTACKS] if args.suite == "all" else SUITE
    )
    if args.categories:
        wanted = {code.strip().upper() for code in args.categories.split(",")}
        pairs = [pair for pair in pairs if pair.category.code in wanted]
    if args.bases:
        wanted = {base.strip().upper() for base in args.bases.split(",")}
        pairs = [pair for pair in pairs if pair.base_id in wanted]
    if args.limit:
        pairs = pairs[: args.limit]
    return pairs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="new_src.adapters.memlineage.run", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", default="memlineage", choices=sorted(CONDITIONS))
    parser.add_argument("--suite", default="core", choices=["core", "licence", "all"])
    parser.add_argument("--categories", default="")
    parser.add_argument("--bases", default="")
    parser.add_argument("--variants", default="H-,H+")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default=ACTION_MODEL)
    parser.add_argument("--kms-url", default=None, help="MemLineage base URL (default $KMS_BASE_URL)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    condition = CONDITIONS[args.condition]
    # The two suites answer different questions; averaging them together produces a number
    # that means nothing, so they never share a records directory (same rule as new_src.run).
    condition_dir = condition.name + ("" if args.suite == "core" else "__licence")
    memlineage = MemLineageClient(args.kms_url) if args.kms_url else MemLineageClient()
    if not memlineage.health():
        print(f"  MemLineage is not reachable at {memlineage.base_url}. Start its backend first.")
        return 2
    workspace = MemLineageWorkspace(memlineage)

    client = make_client()
    pairs = select_pairs(args)
    variants = [value.strip() for value in args.variants.split(",")]
    directory = run_dir(args.model, "module_b", condition_dir)
    records: list = []

    done: set[tuple[str, str]] = set()
    if args.resume:
        for row in load_jsonl(directory / "episodes.jsonl"):
            done.add((row["pair_id"], row["variant"]))
        records.extend(load_jsonl(directory / "episodes.jsonl"))
        print(f"  resume: {len(done)} episodes already recorded")

    for pair in pairs:
        for variant in variants:
            if (pair.pair_id, variant) in done:
                continue
            name = f"{pair.pair_id}_{'minus' if variant == 'H-' else 'plus'}"
            with transcript(args.model, "module_b", condition_dir, name, echo=not args.quiet):
                record = run_episode(client, workspace, pair, variant, condition, args.model, verbose=True)
            append_jsonl(directory / "episodes.jsonl", record)
            records.append(record)

    summary = action_summary([_as_record(r) for r in records])
    lines = [
        f"{'='*72}",
        f"  Module B — condition '{condition.name}' (memory system: MemLineage)",
        f"  {condition.description}",
        f"  endpoint: {BASE_URL}   model: {args.model}   memlineage: {memlineage.base_url}",
        f"  {len(records)} episodes, records appended to {directory / 'episodes.jsonl'}",
        "",
        f"  overall   ASR={_pct(summary['ASR'])}  TSR={_pct(summary['TSR'])}"
        f"   (N-={summary['n_minus']}, N+={summary['n_plus']})",
    ]
    for category, values in by_category([_as_record(r) for r in records], action_summary).items():
        lines.append(f"    {category:<5} ASR={_pct(values['ASR'])}  TSR={_pct(values['TSR'])}")
    path = write_summary(args.model, "module_b", condition_dir, lines)
    print("\n" + "\n".join(lines))
    print(f"\n  summary written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
