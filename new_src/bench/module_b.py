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

from new_src.bench import action_stage, dms, gate
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

    @property
    def name(self) -> str:
        metadata = "meta" if self.show_metadata else "nometa"
        surface = f"-{self.gate_surface}" if self.policy == "gate" else ""
        licence = f"-licence_{self.claim_type_source}" if self.check_license else ""
        return f"{self.policy}{surface}{licence}-{self.label_source}-{self.rendering}-{metadata}" + (
            "-confirm" if self.confirm_followup else ""
        )


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
GATE_PREDICTED = Condition(policy="gate", label_source="predicted", rendering="source_attributed")
GATE_NATIVE = Condition(policy="gate", label_source="gold", rendering="source_attributed", gate_surface="native")
GATE_NATIVE_PREDICTED = Condition(policy="gate", label_source="predicted", rendering="source_attributed", gate_surface="native")
GATE_HEURISTIC = Condition(policy="gate", label_source="heuristic", rendering="source_attributed")
MEMORY_OFF = Condition(policy="direct", label_source="gold", rendering="off")
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

    gate.reset_pending()
    stored = [] if condition.rendering == "off" else records
    if drop_focal:
        from dataclasses import replace

        stored = [
            replace(record, object_ref=None) if record.bound_by_focal else record
            for record in stored
            if not record.is_focal
        ]
    engine, stored = dms.install(episode, stored)
    if condition.check_license and condition.label_source != "channel-typed":
        # The frozen-role conditions still record the live request, as they always did.
        #
        # Under the channel model it is deliberately NOT recorded. It would be `user`, not a
        # quotation, therefore `authorized`, and the write path could well list the very action
        # being attempted against it — while the paired episodes hold that request IDENTICAL
        # across H- and H+, so it cannot be what distinguishes them. Letting it license would
        # blind the gate on every licensing transition at once.
        #
        # The rule it stands for is worth stating plainly: the request being served is not its
        # own warrant. Authorization has to be on record already, or whoever can forge the
        # request can authorize themselves with it.
        dms.capture_user_turn(engine, episode.later_task)

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
