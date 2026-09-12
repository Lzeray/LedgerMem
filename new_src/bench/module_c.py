"""
Module C — end-to-end authority preservation.

The full pipeline, with nothing supplied by the benchmark except the source history:

    consolidation -> automatic authority assignment -> retrieval -> tool action

The consolidator writes whatever memory it writes (Module A's consolidator, unchanged). Each
record is then labeled automatically: a predictor picks the message that primarily supports
the record, and the frozen role policy — not the model — maps that role to a label. The
records go into the store, retrieval pulls what it pulls for the later task, and the action
stage runs exactly as in Module B.

Two things follow from this, both deliberate and both from the paper:

  * All pairs stay in the denominators. If consolidation drops the proposition, or retrieval
    fails to surface it, that shows up as a task failure on H+ — it is not excluded.
  * The operative value is attached to a record by a deterministic retention test (the same
    one Module A's cross-check uses), never by asking a model to extract it. Slot attachment
    is part of the harness, so a labeling result can never be an artifact of a small model's
    extraction — and the value the gate then reads is the benchmark's canonical one.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from new_src.bench import action_stage, dms, gate, module_a
from new_src.bench.actions import TARGET_ACTIONS, tool_trust
from new_src.bench.authority import predict_label
from new_src.bench.classifier import action_catalogue, decide
from new_src.bench.metrics import ActionRecord, mentions_value
from new_src.bench.taxonomy import channel_for
from new_src.bench.schema import AuthorityPair, MemoryRecord, Variant
from new_src.config import ACTION_MODEL
from new_src.memory import recall_facts


def _to_records(client, model, episode, pair, consolidated: list[str], label_source: str) -> list[MemoryRecord]:
    """Turn the consolidator's free text into labeled records.

    Under `channel-typed` this is a two-step decision and both steps are constrained. The
    predictor says which MESSAGE primarily supports the record — a question about the
    conversation, not about authority — and the channel then follows from that message's role
    plus, for a tool result, the declared trust of the tool the episode actually called. Only
    then does the classifier choose a speech act within what that channel permits, and narrow
    the label if the source was relaying somebody else. The model never names a label.

    `object_ref` is attached by the harness, from a deterministic test of whether the record
    names the object the action operates on — asking a model which records are "about" an
    account would put the binding back in its hands, which is the hole license_attacks family A
    exists to exhibit.

    `verbatim` stays empty here, and deliberately so: consolidated text IS a paraphrase, and a
    record holding only a paraphrase cannot authorize anything. That is not a limitation of this
    function — it is the end-to-end finding. What makes Module C measurable anyway is
    `_capture_history`, which stores the utterances themselves alongside these summaries, so
    consolidation keeps deciding the ARGUMENT path while authorization rests on evidence that
    survived.
    """
    object_value = pair.object_value
    records: list[MemoryRecord] = []
    for text in consolidated:
        # Same retention test Module A's cross-check uses, so a consolidator that reworded
        # the claim still counts as having kept it, while a changed value does not.
        carries_value = mentions_value(text, pair.operative_value)
        # Licensing transitions have no argument slot to attach to (see schema.py).
        slotted = carries_value and not pair.category.licenses_action
        draft = MemoryRecord(
            text=text, label="unendorsed", role="tool", rendering="source_attributed",
            slot_key=pair.slot_key if slotted else None,
            slot_value=pair.operative_value if slotted else None,
            is_focal=carries_value,
        )
        channel = claim_type = None
        if label_source == "predicted":
            label, role = predict_label(client, model, episode, draft)
        elif label_source == "gold":
            label, role = (episode.focal_memory.label, episode.focal_memory.role) if carries_value else ("attested", "assistant")
        elif label_source == "channel-typed":
            _, role = predict_label(client, model, episode, draft)
            channel = (channel_for("tool", tool_trust(pair.source_tool.name)) if role == "tool"
                       else channel_for(role))
            decision = decide(client, model, channel, text, action_catalogue())
            label, claim_type = decision.label, None
        else:
            raise ValueError(f"module C does not support label source {label_source!r}")
        records.append(
            MemoryRecord(text=text, label=label, role=role, rendering="source_attributed",
                         slot_key=draft.slot_key, slot_value=draft.slot_value, is_focal=carries_value,
                         claim_type=claim_type, channel=channel,
                         # Deterministic, like slot attachment: a record is about the object when
                         # it names it. Asking a model which records are "about" the account
                         # would put the binding back in the model's hands, which is the hole
                         # data/license_attacks.py family A already exists to exhibit.
                         object_ref=object_value if (object_value and object_value in text) else None)
        )
    return records


def _capture_history(client, model, engine, episode, condition) -> None:
    """Store every utterance of the episode verbatim, beside whatever consolidation produced.

    This is the write path a real agent should have: the moment a message arrives, copy what was
    said, record which channel it came on, classify the act from those words, and take the label
    from the table. None of that needs a consolidator, and all of it is lost if the only thing
    kept is a summary.

    It is what makes Module C able to measure a licensing transition at all. A consolidated
    record is a paraphrase — the consolidator's sentence, not the speaker's — and the licence
    check refuses to authorize from one, on the grounds that a model's choice of wording would
    otherwise decide whether an action is permitted. Before this, every licensing transition in
    Module C was refused for that reason alone, whatever the model did, so the cell measured the
    harness rather than the system.

    Consolidation is NOT bypassed: its records still go in, still carry the operative value's
    slot when they retain it, and still decide the argument path. What changes is that the
    evidence for authorization survives alongside the summary instead of being replaced by it.

    Captured records deliberately carry no slot. They can authorize an action; they can never
    become the value of one of its arguments.
    """
    if condition.label_source != "channel-typed":
        dms.capture_user_turn(engine, episode.later_task)
        return

    spec = TARGET_ACTIONS.get(episode.target_tool)
    scope_param = getattr(getattr(spec, "requires_license", None), "scope_param", None)
    object_value = episode.target_arguments.get(scope_param) if scope_param else None

    pending_tool: str | None = None
    for message in episode.messages:
        if message.tool_call is not None:
            pending_tool = message.tool_call.name
        if not message.content.strip():
            continue
        dms.capture(engine, client, model, message.role, message.content,
                    tool_name=pending_tool if message.role == "tool" else None,
                    object_value=object_value)
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
    consolidated = module_a.consolidate(client, episode)
    written = _to_records(client, model, episode, pair, consolidated, condition.label_source)

    # The customer's identity is a bank-owned record, not something consolidation produces;
    # it is seeded in every condition so each action has exactly one authorized argument and
    # one argument that depends on the transition.
    written = [*episode.context_memory, *written]

    gate.reset_pending()
    engine, written = dms.install(episode, written)
    if condition.check_license:
        _capture_history(client, model, engine, episode, condition)

    focal = next((record for record in written if record.is_focal), None)
    with Session(engine) as session:
        retrieved_rows = recall_facts(session, episode.later_task, top_k=6)
    retrieved = [
        MemoryRecord(text=row.fact_text, label=row.label, role=row.role, rendering=row.rendering,
                     slot_key=row.slot_key, slot_value=row.slot_value,
                     is_focal=row.slot_key == pair.slot_key,
                     claim_type=row.claim_type, object_ref=row.object_ref,
                     verbatim=row.verbatim, channel=row.channel, record_id=row.id)
        for row in retrieved_rows
    ]

    if verbose:
        print(f"\n{'='*72}\n  {pair.pair_id}  {variant}  [module C: {condition.name}]  target={episode.target_tool}")
        print(f"  consolidated : {len(consolidated)} record(s)")
        if focal is None:
            print("  focal claim  : NOT RETAINED by consolidation (write-time omission)")
        else:
            print(f"  focal record : {focal.text}")
            print(f"  label        : {focal.label} (predicted role {focal.role}, channel {focal.channel})"
                  f"  claim_type={focal.claim_type}  verbatim={'yes' if focal.verbatim else 'no'}")
        print(f"  retrieved    : {sum(record.is_focal for record in retrieved)} of {len(retrieved)} rows carry the operative value")

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
    return action_stage.perform(client, episode, engine, retrieved, condition, record, model, verbose)
