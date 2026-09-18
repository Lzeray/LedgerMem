"""
Module C — end-to-end authority preservation.

The full pipeline, with nothing supplied by the benchmark except the source history:

    consolidation -> automatic authority assignment -> retrieval -> tool action

**The system under test owns its memory end to end.** It is given the conversation and
nothing else — no slot keys, no operative values, no target arguments, no object, no category.

  * Write. The consolidator (Module A's, the paper's objective, unchanged) writes whatever
    memories it writes. The memory system labels each one, extracts the operative values it
    states into slots (`bench/slots.py`, a model call bounded by a closed key set and a
    literal-occurrence check), and derives the object it names from those slots. Under the
    channel model it also captures every utterance verbatim as it arrives (`_capture_history`),
    with the same extraction.
  * Retrieve. As in the paper (appendix F.2): "a deterministic bounded in-memory store that
    returns the complete product-generated write set in stable order". The agent is shown every
    consolidated memory, in the order the consolidator wrote them. The verbatim records
    `_capture_history` adds are the gate's evidence store and are never shown to the agent: the
    paper's write set is the consolidator's output, and nothing else.
  * Act. The action stage is Module B's, unchanged — the paper's instruction, its memory
    block, and only the first native action scored — so the two modules' ASR/TSR stay
    comparable in everything except where the memory came from.

What the benchmark still knows, and only for SCORING after the run: the action predicate
(`episode.target_tool`, `episode.target_arguments`) and which record carries the contested
value (`is_focal`, by the deterministic retention test). Neither is shown to or read by the
system. Nothing in the write or read path reads the answer key — including the Oracle arm,
whose reference labels come from a model, as in the paper.

An earlier version attached `slot_key`/`slot_value` from the dataset, set `object_ref` from
`target_arguments`, and wrote the customer's identifiers as separate harness-built records
with the request classifier switched off. Each of those handed the system part of the answer.
That version is tagged `module-c-oracle-slots` in git; its numbers are not comparable with
this one.

All pairs stay in the denominators. If consolidation drops the proposition, extraction misses
its value, or the agent does not use it, that shows up as a task failure on H+.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from new_src.bench import action_stage, dms, gate, module_a
from new_src.bench.authority import predict_sources
from new_src.bench.classifier import action_catalogue, decide
from new_src.bench.metrics import ActionRecord, mentions_value
from new_src.bench.schema import AuthorityPair, MemoryRecord, Variant
from new_src.bench.taxonomy import label_for_role, worst_label
from new_src.bench.slots import extract_slots, object_ref_for
from new_src.config import ACTION_MODEL, JUDGE_MODEL


def _source_roles(episode) -> list[str]:
    """Roles of the messages that can source a durable memory: every non-system message with
    content. The paper: "The system role supplies policy context and cannot be the direct
    source of a durable item"."""
    return [m.role for m in episode.messages if m.role != "system" and m.content.strip()]


def _naive_join_label(episode) -> tuple[str, str]:
    """The paper's Naive-join arm (appendix F.2): "every memory receives the most restrictive
    label appearing anywhere in the write window"."""
    roles = _source_roles(episode)
    labels = [label_for_role(role) for role in roles]
    label = worst_label(*labels)
    return label, next(role for role in roles if label_for_role(role) == label)


def _to_records(client, model, episode, pair, consolidated: list[tuple[str, str]], label_source: str) -> list[MemoryRecord]:
    """Turn the consolidator's free text into labeled, slotted records.

    Slots come from `extract_slots` on the record's own text — the system's decision, not the
    dataset's. A record stating several values becomes one row per value (a row holds one slot).

    Labels:
      channel-typed  consolidated text is the AGENT's own writing, whoever's claim it is about,
                     so its channel is `assistant` and its ceiling `attested`. Structural; never
                     asked, never predicted.
      predicted      the paper's arm (appendix C.3): one source-first call over the whole write
                     set names each memory's supporting message; the frozen role policy maps
                     that message's role to a label in code.
      reference      the paper's Oracle arm (`gold` is accepted as the same thing): the same
                     source-first attribution, run by the configured judge model.
      naive-join     the paper's Naive-join arm: the write window's most restrictive label.

    `verbatim` stays empty: consolidated text IS a paraphrase, and a paraphrase cannot license
    anything. Authorization rests on `_capture_history`'s verbatim records instead.
    """
    records: list[MemoryRecord] = []
    predicted = None
    if label_source == "predicted":
        predicted = predict_sources(client, model, episode, consolidated)
    elif label_source in ("reference", "gold"):
        # The Oracle arm. The paper's reference labels come from a separate, fixed model
        # labeler (GPT-5.6-Luna, appendix C.4) whose prompt it does not print; here that role is
        # played by the configured judge model, running the one published attribution prompt
        # (C.3) at temperature zero. It reads nothing from the dataset's answer key. When the
        # judge and the action model are the same checkpoint, Oracle and Predicted coincide —
        # set AUTHMEM_JUDGE_MODEL to the strongest model available to separate them.
        predicted = predict_sources(client, JUDGE_MODEL, episode, consolidated)
    for position, (text, _memory_type) in enumerate(consolidated):
        # Scoring only: which record carries the contested value. Not stored as a slot, not
        # shown to the system, not read by the gate. The gold arm uses it by definition.
        carries_value = mentions_value(text, pair.operative_value)
        channel = None
        requests = None
        if label_source == "predicted":
            label, role = predicted[position]
        elif label_source in ("reference", "gold"):
            label, role = predicted[position]
        elif label_source == "naive-join":
            label, role = _naive_join_label(episode)
        elif label_source == "channel-typed":
            # See the pinned rule in the project memory: the source of a record is never
            # reconstructed by a model afterwards. Consolidated text was written by the agent.
            role, channel = "assistant", "assistant"
            decision = decide(client, model, channel, text, action_catalogue())
            label, requests = decision.label, decision.requests
        else:
            raise ValueError(f"module C does not support label source {label_source!r}")

        slots = extract_slots(client, model, text)
        object_ref = object_ref_for(slots)
        for slot_key, slot_value in (slots or [(None, None)]):
            records.append(
                MemoryRecord(text=text, label=label, role=role, rendering="source_attributed",
                             slot_key=slot_key, slot_value=slot_value, is_focal=carries_value,
                             claim_type=None, channel=channel, requests=requests,
                             object_ref=object_ref)
            )
    return records


def _capture_history(client, model, engine, episode, condition) -> None:
    """Store every utterance of the episode verbatim, beside whatever consolidation produced.

    This is the write path a real agent should have: the moment a message arrives, copy what was
    said, record which channel it came on, label it from that channel, note what it asks for and
    which values it states. `dms.capture` does all of it from the message alone; nothing here
    reads the pair.

    It is the gate's evidence store: a consolidated record is a paraphrase written by the agent,
    and neither its channel nor its licence can be trusted. The system message is not captured:
    system policy is not a durable memory source (paper, appendix C.3).

    The customer's identifiers are part of the first message of the history (see
    `AuthorityPair._identity_preamble`) and go through this path like everything else they
    said — including the request classifier, which used to be switched off for them. That
    switch hid a real classifier failure ("My portfolio is PF-…" read as a request to
    re-allocate it), and it is now measured instead of suppressed.
    """
    if condition.label_source != "channel-typed":
        return

    pending_tool: str | None = None
    for message in episode.messages:
        if message.tool_call is not None:
            pending_tool = message.tool_call.name
        if not message.content.strip() or message.role == "system":
            continue
        dms.capture(engine, client, model, message.role, message.content,
                    tool_name=pending_tool if message.role == "tool" else None)
    # The closing request is deliberately NOT recorded — see the note in module_b.run_episode.
    # The request being served is not its own warrant.


def run_episode(
    client,
    pair: AuthorityPair,
    variant: Variant,
    condition,
    model: str = ACTION_MODEL,
    verbose: bool = True,
) -> ActionRecord:
    episode = pair.episode(variant)
    consolidated = module_a.consolidate_items(client, episode)
    written = _to_records(client, model, episode, pair, consolidated, condition.label_source)

    # Nothing is seeded. The paper's Module C starts from the source history alone; the
    # customer's identifiers are in the history, so consolidation and capture have to keep
    # them like anything else, and a dropped identity counts against the run.
    gate.reset_pending()
    engine, written = dms.install(episode, written)
    if condition.check_license:
        _capture_history(client, model, engine, episode, condition)
    if condition.policy == "gate":
        # The fixed argument the customer states in q reaches the gate's store as their own
        # words, never licensing anything (see dms.capture_live_request).
        dms.capture_live_request(engine, client, model, episode.later_task)

    focal = next((record for record in written if record.is_focal), None)

    if verbose:
        print(f"\n{'='*72}\n  {pair.pair_id}  {variant}  [module C: {condition.name}]  target={episode.target_tool}")
        print(f"  consolidated : {len(consolidated)} record(s)")
        if focal is None:
            print("  focal claim  : NOT RETAINED by consolidation (write-time omission)")
        else:
            print(f"  focal record : {focal.text}")
            print(f"  label        : {focal.label} (role {focal.role}, channel {focal.channel})"
                  f"  slot={focal.slot_key}={focal.slot_value}")
        with Session(engine) as session:
            from new_src.memory import all_facts

            for row in all_facts(session):
                if row.slot_key:
                    print(f"  slot         : {row.slot_key}={row.slot_value}  [{row.label}, {row.channel or row.role}]")

    record = ActionRecord(
        module="C", pair_id=pair.pair_id, base_id=pair.base_id, category=pair.category.code,
        variant=variant, policy=condition.policy, label_source=condition.label_source,
        rendering="consolidated", show_metadata=condition.show_metadata,
        gate_surface=condition.gate_surface,
        action_permitted=episode.action_permitted, target_tool=episode.target_tool,
        target_arguments=episode.target_arguments, performed=False,
        focal_label_stored=focal.label if focal else "",
        focal_role_stored=focal.role if focal else "",
        notes="" if focal else "write-time omission: focal claim not consolidated",
    )
    # The complete write set, in the consolidator's order (the paper's retrieval) — or nothing,
    # in the Memory-off arm ("frozen writes are not exposed").
    shown = [] if condition.rendering == "off" else written
    return action_stage.perform(client, episode, engine, shown, condition, record, model, verbose)
