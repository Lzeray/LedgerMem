"""
Module B — controlled downstream consequence.

The benchmark hands the action agent the focal memory directly (so consolidation and recall
are not part of what is being measured), together with the later task q and one authority
metadata condition, and watches whether the target action is performed. H- episodes score
ASR (the action was prohibited), H+ episodes score TSR (it was required).

A condition is four choices:

  policy         'direct' — the agent holds the seven banking tools and decides for itself.
                 'gate'   — every action goes through the authority gate, which enforces the
                            label policy in code and never consults the model about it.
  label_source   where each record's authority label comes from: gold annotation, the
                 content-based heuristic baseline, or the Module C role predictor.
  rendering      'source_attributed' keeps the source condition in the record's text;
                 'washed' removes it, leaving the bare proposition; 'off' gives the agent no
                 memory at all (the memory-off baseline).
  show_metadata  whether labels are displayed to the agent in its context.

`direct` + no metadata + washed is the paper's collapsed condition, and is what the
unprotected baseline run uses. `gate` conditions are the defense.
"""

from __future__ import annotations

from dataclasses import dataclass

from new_src.bench import action_stage, dms
from new_src.bench.actions import TARGET_ACTIONS
from new_src.bench.authority import apply_labels
from new_src.bench.metrics import ActionRecord
from new_src.bench.schema import AuthorityPair, Variant
from new_src.config import ACTION_MODEL

@dataclass(frozen=True)
class Condition:
    policy: str = "direct"
    label_source: str = "gold"
    rendering: str = "source_attributed"
    show_metadata: bool = False
    #: How the gate is presented to the model. 'gateway' is the single-tool design: one
    #: protected-action tool plus a question-asking tool. 'native' keeps the seven ordinary
    #: banking tool schemas and routes every call through the gate underneath, ignoring the
    #: arguments the model supplied and resolving them from labeled memory instead. Both
    #: enforce the identical policy; they differ only in the tool surface the model sees,
    #: which on small local models is worth a large amount of tool-calling reliability.
    gate_surface: str = "gateway"
    #: Run the gate's step 0 — is this action authorized at all — before resolving arguments.
    check_license: bool = False
    #: How a record's claim type is decided: "role" maps the source role by a frozen mapping,
    #: "declared" takes the type the taxonomy states for the transition's carrier, and "model"
    #: leaves it to the classifier in bench/classifier.py, constrained to the set its channel
    #: permits. "model" belongs with label_source="channel-typed": both halves of the
    #: decision come from the same classification, and pairing "model" with a role-derived
    #: label would leave the label axis stipulated while the type axis is not.
    claim_type_source: str = "role"
    #: Beyond the paper: run one scripted confirmation round-trip after the gate asks for it.
    #: Off by default, because the paper's ASR/TSR are measured without a user in the loop.
    confirm_followup: bool = False
    #: Beyond the paper: the agent is given no memory block at all and must fetch what it needs
    #: through a search tool (see action_stage.SEARCH_TOOL_NAME). The store then holds every
    #: focal record of this customer's pairs in the suite, so retrieval can miss. This is where
    #: the gate's shape shows: it resolves arguments from the store itself, so a retrieval miss
    #: costs it nothing, while a direct agent has to find the value before it can use it.
    retrieval: bool = False
    #: Module C only. `True`: the agent is shown the write-time journal (every message of the
    #: conversation, verbatim, as "role: text", no labels) instead of the consolidated memory —
    #: the same complete record the gate reads, so a direct agent given it isolates what complete
    #: memory is worth from what labels and the gate add.
    journal: bool = False
    #: Retrieval arms: "store" searches every row the system holds (consolidated memory and, for
    #: the gate, its journal); "shown" searches only the memory an unprotected agent would get.
    search_scope: str = "store"
    #: The paper's text-sanitizer baseline (appendix C.2): the washed item plus only a fixed
    #: warning that memory may be distorted or unreliable.
    sanitizer: bool = False

    @property
    def name(self) -> str:
        metadata = "meta" if self.show_metadata else "nometa"
        surface = f"-{self.gate_surface}" if self.policy == "gate" else ""
        licence = f"-licence_{self.claim_type_source}" if self.check_license else ""
        return f"{self.policy}{surface}{licence}-{self.label_source}-{self.rendering}-{metadata}" + (
            "-retrieve" if self.retrieval else "") + (
            "-consolidatedonly" if self.search_scope == "shown" else "") + (
            "-journal" if self.journal else "") + (
            "-sanitize" if self.sanitizer else ""
        ) + ("-confirm" if self.confirm_followup else "")


# ---- The paper's seven Module-B interventions (appendix E.1) -------------------------------
#   Off       MEMORY_OFF          no memory
#   W/N       BASELINE            washed, no metadata
#   S/N       BASELINE_ATTRIBUTED source-attributed, no metadata
#   Sanitize  SANITIZER           washed + generic warning
#   W/Join    CONSERVATIVE_JOIN   washed, Unendorsed for both variants
#   W/G       GOLD_WASHED         washed, gold role-derived label
#   S/G       GOLD_PROMPTED       source-attributed, gold role-derived label
# Everything with policy="gate" is this project's defense, not a paper condition.

# The unprotected baseline: washed memory, no authority metadata, tools in the agent's hands.
BASELINE = Condition(policy="direct", label_source="gold", rendering="washed", show_metadata=False)
# Same agent, but records still name their source in prose (no machine-readable labels).
BASELINE_ATTRIBUTED = Condition(policy="direct", label_source="gold", rendering="source_attributed")
# Metadata given to the model, which must apply the policy itself.
GOLD_PROMPTED = Condition(policy="direct", label_source="gold", rendering="source_attributed", show_metadata=True)
HEURISTIC_PROMPTED = Condition(policy="direct", label_source="heuristic", rendering="source_attributed", show_metadata=True)
# The same content-based guess applied to washed records, where by construction there is no
# source condition left to read. Offline, its labels are right on exactly half the pairs.
HEURISTIC_WASHED = Condition(policy="direct", label_source="heuristic", rendering="washed", show_metadata=True)
# The defense: enforcement in code, from stored labels.
GATE_GOLD = Condition(policy="gate", label_source="gold", rendering="source_attributed")
GATE_GOLD_WASHED = Condition(policy="gate", label_source="gold", rendering="washed")
# The paper's own Module C arm, end to end: consolidation writes the records, a predictor
# picks the message that primarily supports each one, and the frozen role policy maps that
# role to a label — "we then map its source role deterministically to Authorized, Attested or
# Unendorsed". No gate, no metadata shown, no seeded memory, no verbatim capture. This is the
# unprotected reproduction every extension in this project has to be measured against.
BASELINE_PREDICTED = Condition(policy="direct", label_source="predicted",
                               rendering="source_attributed")
GATE_NATIVE = Condition(policy="gate", label_source="gold", rendering="source_attributed", gate_surface="native")
MEMORY_OFF = Condition(policy="direct", label_source="gold", rendering="off")
SANITIZER = Condition(policy="direct", label_source="gold", rendering="washed", sanitizer=True)
CONSERVATIVE_JOIN = Condition(policy="direct", label_source="conservative-join", rendering="washed",
                              show_metadata=True)
GOLD_WASHED = Condition(policy="direct", label_source="gold", rendering="washed", show_metadata=True)

# ---- The paper's five Module-C conditions (appendix F.2) ----------------------------------
#   Memory off  MEMORY_OFF   (module C honours rendering="off")
#   No label    C_NO_LABEL   every written memory as plain text
#   Naive join  C_NAIVE_JOIN every memory gets the most restrictive label in the write window
#   Predicted   C_PREDICTED  the source-first predictor's role-derived label
#   Oracle      C_ORACLE     the reference label
C_NO_LABEL = Condition(policy="direct", label_source="reference", rendering="source_attributed")
C_NAIVE_JOIN = Condition(policy="direct", label_source="naive-join", rendering="source_attributed",
                         show_metadata=True)
C_PREDICTED = Condition(policy="direct", label_source="predicted", rendering="source_attributed",
                        show_metadata=True)
C_ORACLE = Condition(policy="direct", label_source="reference", rendering="source_attributed",
                     show_metadata=True)
# Beyond the paper: the agent retrieves memory itself instead of being handed it.
BASELINE_RETRIEVE = Condition(policy="direct", label_source="gold", rendering="washed", retrieval=True)
# The same for the speech-act suites, which define no washed rendering (their unprotected arm is
# baseline-attributed): the agent searches source-attributed records instead.
BASELINE_ATTRIBUTED_RETRIEVE = Condition(policy="direct", label_source="gold", rendering="source_attributed",
                                         retrieval=True)
GATE_RETRIEVE = Condition(policy="gate", label_source="channel-typed", rendering="source_attributed",
                          check_license=True, claim_type_source="model", retrieval=True)
# Module C's unprotected arm with retrieval: the agent searches the memory the system consolidated
# itself (plain text, no labels) instead of being shown it. `gate-retrieve` serves both modules.
C_NO_LABEL_RETRIEVE = Condition(policy="direct", label_source="reference", rendering="source_attributed",
                                retrieval=True)
# Module C controls for what complete memory is worth. `c-journal`: an unprotected agent shown the
# whole write-time journal instead of the consolidated memory. `gate-retrieve-consolidated`: the
# gate as before (it still reads its journal), but the agent's own search covers only the
# consolidated memory, as an unprotected agent's does.
C_JOURNAL = Condition(policy="direct", label_source="reference", rendering="source_attributed", journal=True)
GATE_RETRIEVE_CONSOLIDATED = Condition(policy="gate", label_source="channel-typed", rendering="source_attributed",
                                       check_license=True, claim_type_source="model", retrieval=True,
                                       search_scope="shown")

# The gate with action-level authorization added, in both claim-type modes.
GATE_LICENSE = Condition(policy="gate", label_source="gold", rendering="source_attributed",
                         check_license=True, claim_type_source="role")
GATE_LICENSE_DECLARED = Condition(policy="gate", label_source="gold", rendering="source_attributed",
                                  check_license=True, claim_type_source="declared")
# The channel model: the label's ceiling and the permitted speech acts come from the channel a
# claim arrived on, and a classifier chooses within that. This is the condition the four
# speech-act families (Q2D, N2D, P2F, G2O) were written to discriminate, and the only one in
# which a claim type is not a function of the source's role.
GATE_LICENSE_MODEL = Condition(policy="gate", label_source="channel-typed",
                               rendering="source_attributed", check_license=True,
                               claim_type_source="model")


def _other_records(pair, condition) -> list:
    """The focal records of this customer's other pairs, as distractors for the retrieval arm."""
    from dataclasses import replace

    from new_src.data.heldout import HELDOUT_SUITE
    from new_src.data.multiarg import MULTIARG_DEV, MULTIARG_HELDOUT
    from new_src.data.speech_act_attacks import SPEECH_ACT_DEV, SPEECH_ACT_HELDOUT_V2
    from new_src.data.suite import SUITE

    others = []
    for suite in (SUITE, HELDOUT_SUITE, MULTIARG_DEV, MULTIARG_HELDOUT, SPEECH_ACT_DEV, SPEECH_ACT_HELDOUT_V2):
        if not any(other.pair_id == pair.pair_id for other in suite):
            continue
        for other in suite:
            if other.base_id != pair.base_id or other.pair_id == pair.pair_id:
                continue
            record = other.episode("H+").focal_memory
            others.append(replace(record, is_focal=False))
    return others


def run_episode(
    client,
    pair: AuthorityPair,
    variant: Variant,
    condition: Condition,
    model: str = ACTION_MODEL,
    module: str = "B",
    verbose: bool = True,
    drop_focal: bool = False,
) -> ActionRecord:
    """`drop_focal` is the null control: run the episode with the contested record removed and
    everything else untouched. The action must NOT fire. If it fires anyway, the later task
    alone was enough to trigger it and the pair proves nothing about the authority transition —
    the same trap as a scenario that cannot fail, in the opposite direction."""
    rendering = "source_attributed" if condition.rendering == "off" else condition.rendering
    episode = pair.episode(variant, rendering=rendering, claim_type_source=condition.claim_type_source)
    records = apply_labels(client, model, episode, condition.label_source)
    focal = next(record for record in records if record.is_focal)

    stored = [] if condition.rendering == "off" else records
    if drop_focal:
        from dataclasses import replace

        stored = [
            replace(record, object_ref=None) if record.bound_by_focal else record
            for record in stored
            if not record.is_focal
        ]
    if condition.retrieval:
        # Everything this customer has on file in the suite, so the search has something to miss:
        # one focal record per other pair of the same base. Their slot keys differ by transition,
        # so they add noise to retrieval without making any argument ambiguous.
        stored = [*stored, *_other_records(pair, condition)]
    engine, stored = dms.install(episode, stored)
    if condition.policy == "gate":
        # The gate reads memory, never the conversation, and Module B's memory is the single
        # focal item — so the fixed argument the customer states in q has to reach the store
        # the way any live user turn would: captured as the customer's own words, labeled and
        # classified like every other utterance, with its values extracted by the memory system.
        # It is identical in H- and H+.
        dms.capture_live_request(engine, client, model, episode.later_task)

    if verbose:
        print(f"\n{'='*72}\n  {pair.pair_id}  {variant}  [{condition.name}]  target={episode.target_tool}")
        print(f"  focal record : {focal.text}")
        print(f"  stored label : {focal.label} (role {focal.role}, channel {focal.channel})"
              f"  claim_type={focal.claim_type}  verbatim={'yes' if focal.verbatim else 'no'}")
        print(f"  later task   : {episode.later_task}")

    record = ActionRecord(
        module=module, pair_id=pair.pair_id, base_id=pair.base_id, category=pair.category.code,
        variant=variant, policy=condition.policy, label_source=condition.label_source,
        rendering=condition.rendering, show_metadata=condition.show_metadata,
        gate_surface=condition.gate_surface,
        action_permitted=episode.action_permitted, target_tool=episode.target_tool,
        target_arguments=episode.target_arguments, performed=False,
        # Nothing was stored in the memory-off condition, and nothing but context in the null
        # control, so recording a focal label for either would misdescribe the run.
        focal_label_stored="" if (drop_focal or condition.rendering == "off") else focal.label,
        focal_role_stored="" if (drop_focal or condition.rendering == "off") else focal.role,
        notes="null control: contested record removed" if drop_focal else (
            "memory-off: no records available to the agent" if condition.rendering == "off" else ""),
    )
    return action_stage.perform(client, episode, engine, stored, condition, record, model, verbose)
