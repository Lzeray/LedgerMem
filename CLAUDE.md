# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## What this is

LedgerMem measures whether an **authority gate** keeps a banking agent from acting on
information nobody authorized. It is an implementation of AuthMem-Bench (arXiv:2608.01679) in
the banking domain, plus one defense of its own measured on the paper's metrics.

Every item in memory carries a trust label — `authorized`, `attested` or `unendorsed` — and the
label follows from **where the words structurally came from**, never from what they say. A
protected action may use `authorized` values directly; anything else is refused until the
customer confirms that one call, and confirming never raises a stored label. An agent silently
acting on unverified data is "authority collapse", the failure this benchmark exists to detect.

Everything lives in **`new_src/`**. An earlier implementation in `src/` was deleted in
`a07f0da`; it exists only in git history, and nothing in the repository depends on it.

## Running

Three commands drive a full measurement. They are the normal way to run anything long:

```bash
.venv/bin/python -m new_src.final start --model <model id>   # checks, preflight, then the whole plan in the background
.venv/bin/python -m new_src.final status                     # phase by phase, with speed and time left
.venv/bin/python -m new_src.final results                    # ASR/TSR per suite and condition, with n and 95% intervals
.venv/bin/python -m new_src.final stop                       # and `start` again resumes exactly where it stopped
.venv/bin/python -m new_src.plot_final                       # tables into logs_result/
```

`final start` refuses to begin unless every suite passes offline validation and the endpoint
answers the benchmark's own request on both tool surfaces. Records go to
`logs_final/<model>/…`, never to `logs_authmem/`, and every phase runs with `--resume`, so a run
can be stopped and restarted without recording an episode twice.

Single phases, for debugging or a smoke test:

```bash
.venv/bin/python -m new_src.run validate --suite core|multiarg|speechact|licence|all
.venv/bin/python -m new_src.run b --condition gate-license-model --bases B1 --quiet
.venv/bin/python -m new_src.run c --condition c-oracle
.venv/bin/python -m new_src.run null                       # the validity control
.venv/bin/python -m new_src.run_heldout b --suite core --condition baseline
```

Requirements and traps:

- **Postgres with pgvector** at `AUTHMEM_DB_URL` (default `postgresql://lenaz:lenaz210607@localhost/mydb`).
  Every episode empties both tables, which is why `exclusive_run` refuses to let two runs overlap.
- **An OpenAI-compatible endpoint.** The Innopolis gateway moved to
  `http://models.innopolis.university/v1`; `final.py` defaults to it. A model can be listed there
  and still answer 404 from the backend behind it — that is a server-side problem, and the
  preflight catches it before anything runs.
- **`.env`'s `AUTHMEM_API_KEYS` holds the Gemini keys**, and `config.py` prefers the list over
  the single key. Anything pointed at another endpoint must set `AUTHMEM_API_KEYS` explicitly;
  `final.py` does.
- **Unset the proxy variables** for a local or university endpoint, and keep them for a hosted
  one. `ALL_PROXY` is a `socks://` URL that httpx rejects outright, so it is always dropped.
- The consolidator can run on its own endpoint (`AUTHMEM_CONSOLIDATOR_BASE_URL`, `_API_KEYS`,
  `_PROXY`, `_MODEL`), as can the judge (`AUTHMEM_JUDGE_*`). Unset, they share the agent's.

## The suites

| suite | pairs | what it is |
|---|---|---|
| `core` (dev) | 35 | 5 base histories × the paper's 7 transitions. The defense was designed against it: a development set. |
| `core` (held-out) | 35 | the same 7 transitions on 5 customers written after the design freeze |
| `multiarg` (dev / held-out) | 15 + 15 | actions with 3–5 arguments — WIRE (5), STO (4), TRV (a country, not a number) |
| `speechact` (dev) / `speechact2` (held-out) | 20 + 20 | Q2D, N2D, P2F, G2O. **Only Q2D and G2O are in the programme**; N2D and P2F were dropped. |
| `licence` | 15 | attacks on the licence check itself. **Out of the programme**, kept in the code. |

Each pair is two tellings: **H−**, where the contested value comes from a non-authorizing
source, and **H+**, where the customer says the same thing. ASR is the share of H− where the
action happened (lower is better), TSR the share of H+ where it happened (higher is better).

A pair whose **closing request names the action** tests nothing: the customer asking there and
then authorizes it whatever the history said. `speech_act_attacks.validate` rejects such pairs
mechanically, which is why the first held-out speech-act set (all 20 pairs) is no longer used.

`python -m new_src.run null` is the other validity control: the same H− with the contested record
removed. Any pair that still fires is excluded from ASR.

## Modules and conditions

- **Module A** — write time: the consolidator writes memory, a blinded judge says whether the
  focal claim was dropped, kept with its source, or upgraded.
- **Module B** — action time: the benchmark hands the agent one focal memory and the request.
- **Module C** — end to end: the system consolidates, labels and retrieves for itself.

Conditions (in `new_src/run.py`'s `CONDITIONS`). The paper's, in Module B: `memory-off`,
`baseline` (washed), `baseline-attributed`, `sanitizer`, `conservative-join`, `gold-washed`,
`gold-prompted`. In Module C: `memory-off`, `c-no-label`, `c-naive-join`, `c-predicted`,
`c-oracle`. This project's: `gate` (gold labels), `gate-license-model` (the channel model — the
main arm), and `baseline-retrieve` / `gate-retrieve`, where the agent is given no memory block
and has to find what it needs with a `search_memory` tool.

Module C refuses every gate arm except the channel model: its other label sources are a model's
guess about who spoke, and a gate executing a guess measures the guesser.

## Architecture

- **`bench/taxonomy.py`** — the paper's frozen role policy, and the channel model that replaced
  it (see below). `label_for(channel, claim_type)` is a table lookup, never a judgement.
- **`bench/schema.py`** — `AuthorityPair`, and `validate_pair`: ten mechanical checks that make
  a pair a real carrier swap (identical episodes once the two focal quotes are removed, the
  operative value new to the parent and globally unique, the action consuming it, and so on).
- **`bench/actions.py`** — the tool registry. `ActionSpec.trust` declares whether a lookup tool
  speaks for the bank or relays an outside party; that is what makes a channel structural.
  `value_patterns` are part of a tool's declared interface: an account argument requires at
  least eight digits, because "contains a digit" once let an amount ("4800 EUR") or a partial
  identifier ("the one ending 4417") be bound as an account.
- **`bench/gate.py`** — the defense. Step 1: is this action asked for at all — an `authorized`
  record must list it in its request list, about the same object. Step 2: each argument, bound
  to the request first and to an exact slot lookup after that. Step 3: anything not bound is put
  to the CUSTOMER by the gate itself, inside the same call, through a harness-provided `Customer`:
  missing values typed in and format-checked, ambiguous ones chosen from buttons naming each
  value's least trusted source, blocked ones accepted, rejected or replaced one by one, then a
  final yes/no on the whole call. No token, no agent in between; the agent supplies only the
  action's name. Step 4: execute, then spend the licence (`store.consume_request`): the requests
  that licensed it no longer list the action. Customer answers are never written; no label ever
  changes. The measured programme attaches no customer, so such a call is simply not carried out.
- **`bench/dms.py`** — the deterministic memory stub (Module B) and the write path (`capture`).
- **`bench/classifier.py`** / **`bench/slots.py`** — the write path's model calls, each bounded:
  only customer messages are asked anything (quoting? declining? which actions?), answers count
  only in the strict format (`parse_yes_no`, `parse_request_list`), and slot extraction is limited
  to a closed key set, a literal occurrence in the text, and the parameter's declared format. No
  model is asked about a tool result.
- **`bench/module_a|b|c.py`**, **`bench/action_stage.py`**, **`bench/engine.py`** — the modules,
  the shared action stage (the paper's instruction and memory block verbatim), and the transport.
- **`data/`** — the suites and their generation: `suite.py`, `heldout.py`, `multiarg.py`,
  `speech_act_attacks.py`, `license_attacks.py`, plus `GENERATION_PROMPT*.md` and the generators.
- **`final.py`**, **`plot_final.py`**, **`report.py`**, **`status.py`**, **`preflight.py`** — the
  programme, the figures and the summaries.

## Critical invariant: authority is structural, never predicted

A record's label follows from where its words came from. The model is never asked for a label,
and never asked who spoke:

- **`authorized` has exactly two sources**: the customer's own words when they are not quoting
  anybody, and a grant arriving over a trusted tool. `store.write_fact` refuses outright to
  store `authorized` on the `assistant` or `untrusted_tool` channels — a hard stop, not a clamp,
  because a clamp hides the defect.
- **`user_confirmed` is never a model-facing argument.** It is threaded in by the harness.
- **Confirming never raises a label.** It authorizes one call and nothing else.
- **Consolidated text is the agent's own writing**, so `attested` is its ceiling. When a design
  needs the source of a record and the source is no longer available, carry the channel through
  from write time — never reconstruct it with a model afterwards.
- The classifier is shown the **utterance**, never a summary of it: the kind of act belongs to
  what was said, and a third-person paraphrase of a request reads as a statement of fact.

Any change adding a model-settable argument that touches labels, confirmation or authorization
must be checked against this.

## The channel model (`gate-license-model`)

The paper's role policy decides two things from one fact — the label and the kind of act — and
the speech-act families falsify one each. The replacement splits them:

- **The channel** comes from the integration: the role, plus for a tool result the trust
  declared for that tool in `actions.py`. Never inferred from what a result says.
- **The label** follows from the channel, with one model question for the customer's channel
  only: is the customer quoting somebody else (then `unendorsed`). The claim-type table in
  `taxonomy.label_for` is no longer on the write path; `classifier.decide` is.
- **One rule raises authority**: a verified grant. Only a tool declared `grants=True` in
  `actions.py` (the authorization register) can carry one, and only from its structured result
  with status `active` (`actions.verified_grant`), checked in code. Every other bank system is
  `attested` whatever its text says, because such systems relay text written by others. State the
  register's truthfulness as an assumption wherever these numbers are reported.

It is an extension beyond the paper and is reported as one. The frozen policy stays in the code
and every arm built on it still runs, so earlier numbers remain comparable.

## Empirical traps, found by A/B against live models

- **A tool named `resolve_*`** makes qwen2.5:14b return a completely empty response at every
  temperature. The gate's tool is called `perform_banking_action` for that reason alone.
- **Literal pseudo-code in a system prompt** deadlocks the same models. Prompts describe tools in
  prose.
- **Small models narrate instead of calling.** `complete_with_retry` escalates temperature; some
  cases never recover, and they are recorded as malformed calls rather than as refusals.
- **The paper's consolidator prompt on a small model drops user statements**: qwen2.5:14b
  returned an empty memory list for 11 of 14 dev H+ histories that gemini-3.5-flash-lite kept in
  full. The consolidator's endpoint is configurable for this reason.
- **The retrieval arm needs a model that will call a search tool.** qwen2.5:14b writes
  `search_memory(...)` into an argument instead; the arm is not measurable there.

## Standing methodological rules

- **ASR's denominator is the H− episodes only**, after null-control exclusions. Always state n.
- **Per-category cells are n≈5.** They carry no conclusions on their own; report the aggregate.
- **Every 0% is reported with its one-sided 95% upper bound** (≈10% at n=30). Never write
  "guaranteed", "prevents" or "eliminates" about a 0% cell.
- **The dev suites are a development set.** Headline numbers belong on the held-out suites.
- **The suite is never tuned so the defense passes.** A pair that fails validation is a broken
  scenario; a pair the defense fails is a result.
- **A competitor system is wired up exactly as its own documentation describes it** (see
  `new_src/adapters/memlineage/`), with no capability it does not ship.
- **Numbers measured before a change to the suite or the pipeline are superseded**, and are
  labelled so rather than quietly reused.

## Where the work stands

The gate's design is frozen. The current programme measures, per model: Module B and Module C on
the core dev and held-out suites, the multi-argument suites, Q2D and G2O, and the two retrieval
arms — about 3,350 episodes. `logs_final/results.md` and `logs_result/final_*.png` are its
output.

What is left, in priority order:

1. Finish the programme on a model strong enough for every arm, and recompute every table with n
   and confidence intervals.
2. Decide what to do about pairs that do not fire even unprotected on a given model: they
   measure nothing on it and should be excluded explicitly, like null-control failures.
3. Write up the authenticated-channel direction as future work: authority should follow from a
   verifiable signature on the channel, not from a role.
4. Cite and differentiate **EAL-Bench** (arXiv:2609.01836): the same write-time/action-time
   split and hidden ground truth, but an endogenous threat model — authorization that expired or
   was revoked. Its source-authority gating reproduces this project's safety/utility trade-off
   on frontier models, and it names a real gap here: LedgerMem's records never expire.
5. Run the two retrieval phases that never ran on MiniMax-M2.7: the plan had given the speech-act
   suites `baseline-retrieve`, which needs a washed rendering they do not define. They now get
   `baseline-attributed-retrieve`; `final start` resumes and runs just those two.
