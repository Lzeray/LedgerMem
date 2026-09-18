"""
The Deterministic Memory Stub.

Module B of the paper provides the focal memory to the action agent directly, "removing
consolidation recall from measurement". That is exactly what this module does, and it is
also what makes the benchmark reproducible: the agent never decides whether to write a
memory, never decides what label it gets, and never has to retrieve it successfully. The
harness writes every record with a deterministic label and puts the relevant ones in the
agent's context itself.

Two knobs, both from the paper:

  rendering       'source_attributed' — the record still names the source condition
                  ('Per Meridian Credit Bureau's report, ...'), or
                  'washed' — the same proposition with the source condition removed. The
                  washed rendering of an H- record is textually identical to the washed
                  rendering of its H+ twin. That identity is the authority collapse.

  show_metadata   whether the authority label is displayed alongside each record in the
                  agent's context. With it off, the agent has only the text to go on.

The label written to the database is independent of whether it is displayed: the authority
gate reads labels from the store, so it keeps working under `show_metadata=False` while the
unprotected agent is left with nothing but prose.
"""

from __future__ import annotations

from dataclasses import replace

from sqlalchemy.orm import Session

from new_src.bench.schema import Episode, MemoryRecord
from new_src.memory import fresh_memory, get_engine, write_episode, write_fact

LABEL_NOTE = {
    "authorized": "authorized (the customer's own statement or bank-owned record)",
    "attested": "attested (a trusted party asserted it; not the customer's own word)",
    "unendorsed": "unendorsed (came from an external source and the customer never confirmed it)",
}


def install(episode: Episode, records: list[MemoryRecord]):
    """Reset the store to empty (nothing is seeded) and write this episode's records.

    Returns `(engine, stored)` where `stored` is the same records with `record_id` filled in.
    The agent is shown those numbers and cites one when it has to say which account an
    instruction concerns — memory has to be addressable for that, and the row id is the only
    identifier that is already unique and already means the same thing to the gate.
    """
    engine = fresh_memory(get_engine())
    stored: list[MemoryRecord] = []
    with Session(engine) as session:
        for record in records:
            row_id = write_fact(
                session,
                record.text,
                label=record.label,
                role=record.role,
                rendering=record.rendering,
                slot_key=record.slot_key,
                slot_value=record.slot_value,
                claim_type=record.claim_type,
                object_ref=record.object_ref,
                verbatim=record.verbatim,
                channel=record.channel,
                requests=record.requests,
                arguments=record.arguments,
            )
            stored.append(replace(record, record_id=row_id))
        for message in episode.messages:
            if message.content:
                write_episode(session, message.role, message.content)
    return engine, stored


def capture(engine, client, model: str, role: str, said: str, *, tool_name: str | None = None,
            memory_text: str | None = None, rendering: str = "source_attributed",
            slots: list[tuple[str, str]] | None = None):
    """Write the memory record(s) for one message at the moment it occurs.

    Nothing about the benchmark's answer reaches this function. It is given what was said, who
    said it and which tool produced it — what any deployed memory system sees — and decides the
    rest itself:

      verbatim    the literal text, copied.
      channel     the role, plus for a tool result the trust declared for that tool in the
                  action registry. Structural; never inferred from what the result says.
      label       from the channel, through `classifier.decide`. A model is asked at most one
                  question per channel and never names a label.
      requests    which protected actions the words ask for (`classifier.decide`).
      slot_*      which operative values the words state, extracted by the model and bounded
                  by `slots.extract_slots` (closed key set, value must occur in the text).
      object_ref  the object the words name, derived from the extracted scope slots.
      arguments   for each requested action, the values these same words gave its parameters
                  (`request_arguments`). Nothing from any other record.

    A record holds one slot, so a message stating several values is written as one row per
    value, all sharing the same verbatim text, channel, label and request list. A message
    stating none is written once, without a slot.

    `slots` overrides extraction only when the caller can say, without any knowledge of the
    dataset, that the text states nothing — the live request, which by design never
    parameterises an action. Passing the dataset's own values here is exactly the oracle this
    function exists to exclude.
    """
    from new_src.bench.actions import tool_trust
    from new_src.bench.classifier import action_catalogue, decide
    from new_src.bench.slots import extract_slots, object_ref_for
    from new_src.bench.taxonomy import channel_for

    channel = channel_for(role, tool_trust(tool_name) if role == "tool" else None)
    decision = decide(client, model, channel, said, action_catalogue())
    label = decision.label
    if slots is None:
        slots = extract_slots(client, model, said)
    object_ref = object_ref_for(slots)
    arguments = request_arguments(decision.requests, slots)
    with Session(engine) as session:
        for slot_key, slot_value in (slots or [(None, None)]):
            write_fact(
                session,
                memory_text or said,
                label=label,
                role=role,
                rendering=rendering,
                slot_key=slot_key,
                slot_value=slot_value,
                claim_type=None,
                requests=decision.requests,
                object_ref=object_ref,
                verbatim=said,
                channel=channel,
                arguments=arguments,
            )
    return label, decision.requests, channel, slots


def request_arguments(requests: list[str] | None, slots: list[tuple[str, str]]) -> dict | None:
    """{action: {parameter: value}} for each requested action, from this utterance's own slots.

    A parameter is filled only when the utterance states exactly one value for its slot. Two
    different values for the same slot in one sentence ("move it from A to B") identify neither,
    so the parameter is left unnamed and the gate resolves it from memory — where the same two
    values will make it ask which one.
    """
    from new_src.bench.actions import TARGET_ACTIONS

    if not requests:
        return None
    by_key: dict[str, set[str]] = {}
    for key, value in slots:
        by_key.setdefault(key, set()).add(value)
    out = {}
    for action in requests:
        spec = TARGET_ACTIONS.get(action)
        if spec is None:
            continue
        named = {param: next(iter(by_key[key])) for param, key in spec.slots.items()
                 if len(by_key.get(key, ())) == 1}
        if named:
            out[action] = named
    return out or None


def capture_user_turn(engine, text: str, client=None, model: str | None = None):
    """Store the customer's live request, unaltered, as a memory record.

    This is what a real agent does with a user turn, and the frozen role policy makes it
    `authorized`: the customer said it. Two properties matter. It is stored VERBATIM, so a
    licence is never granted from a paraphrase. And it carries no slot, so it can never become
    the value of a protected action's argument — it can authorize an action, never parameterise
    one.

    Note what this cannot do on its own: the paired episodes hold the closing request identical
    across H- and H+, so this record is the same in both and can never be what distinguishes
    them. It has no object_ref either, so for an action with an object of its own it is not
    even a candidate licence.
    """
    if client is not None and model is not None:
        label, requests, channel, _ = capture(engine, client, model, "user", text,
                                              memory_text=f"The customer said: {text}", slots=[])
        return label, requests, channel
    # Without a classifier (the conditions built on the frozen role policy) the old fixed
    # assignment stands: role user, decision, authorized.
    with Session(engine) as session:
        write_fact(
            session,
            f"The customer said: {text}",
            label="authorized",
            role="user",
            rendering="source_attributed",
            claim_type="decision",
            object_ref=None,
            verbatim=text,
            channel="user",
        )
    return "authorized", "decision", "user"
