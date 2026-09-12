"""
Pre-run check for a remote inference endpoint.

    python -m new_src.preflight

Three things go wrong when pointing this benchmark at a new OpenAI-compatible server, and all
three are cheap to find now and expensive to find on episode forty:

  1. the endpoint is unreachable, or the key is wrong;
  2. the model id is not what it was written down as — vLLM and Ray Serve serve whatever
     `--served-model-name` says, which is often the full HuggingFace path;
  3. the server rejects, or fails to answer, the exact request Module B makes.

The third point is why the probes below build their request out of the benchmark's OWN code —
`action_stage.build_messages` for the message shape and `ActionSpec.openai_schema` for the
tools — rather than a simplified hand-written request. An earlier version of this file probed
with a tidy two-message prompt and a hand-made tool, passed, and the run then died on its first
episode: Ray Serve rejects two consecutive system messages, which the simplified probe never
sent. A preflight that does not exercise the real path is worse than none, because it grants
confidence it has not earned.

Both tool surfaces are probed. The direct surface (the seven banking actions, used by every
`direct` condition) and the gateway surface (the question tool plus the single protected-action
tool, used by every `gate` condition) have different schemas, and a server can accept one and
reject the other.

Exit code 0 means the endpoint is fit to run against.
"""

from __future__ import annotations

import sys


def _sample_request(surface: str):
    """The exact messages and tools one Module B episode would send."""
    from new_src.bench import action_stage, gate
    from new_src.bench.actions import TARGET_ACTIONS
    from new_src.bench.module_b import BASELINE, GATE_GOLD
    from new_src.data.suite import SUITE

    pair = SUITE[0]
    if surface == "direct":
        condition = BASELINE
        episode = pair.episode("H-", rendering="washed")
        tools = [spec.openai_schema() for spec in TARGET_ACTIONS.values()]
    else:
        condition = GATE_GOLD
        episode = pair.episode("H-")
        tools = [gate.build_ask_tool(), gate.build_gate_tool(list(TARGET_ACTIONS))]
    messages = action_stage.build_messages(episode, list(episode.memory), condition)
    return messages, tools


def probe(client, model: str, surface: str) -> bool:
    """Send one real request. Returns True when the server answered with a tool call."""
    messages, tools = _sample_request(surface)
    print(f"        {surface:8} surface: {len(messages)} message(s) "
          f"({', '.join(m['role'] for m in messages)}), {len(tools)} tool(s)")
    try:
        message = client.chat.completions.create(
            model=model, messages=messages, tools=tools, tool_choice="auto",
            temperature=0.0, max_tokens=500,
        ).choices[0].message
    except Exception as error:  # noqa: BLE001
        print(f"        FAILED: {type(error).__name__}: {str(error)[:400]}")
        return False

    calls = getattr(message, "tool_calls", None)
    if not calls:
        print("        FAILED: the model answered in prose and emitted no tool call.")
        print(f"        reply: {(message.content or '')[:200]}")
        print("        Every ASR and TSR here would be 0% for that reason alone, not because")
        print("        of any defense. Module B measures whether a structured call was made.")
        return False
    print(f"        ok — called {calls[0].function.name}({calls[0].function.arguments})")
    return True


def main() -> int:
    from new_src.bench.engine import make_client
    from new_src.config import ACTION_MODEL, BASE_URL, CONSOLIDATOR_MODEL, JUDGE_MODEL

    print(f"  endpoint : {BASE_URL}")
    print(f"  models   : action={ACTION_MODEL}  consolidator={CONSOLIDATOR_MODEL}  judge={JUDGE_MODEL}")
    client = make_client()

    print("\n  [1/3] listing models ...")
    try:
        served = [m.id for m in client.models.list().data]
    except Exception as error:  # noqa: BLE001
        print(f"        FAILED: {type(error).__name__}: {error}")
        print("        The endpoint is unreachable or the key was rejected.")
        return 1
    for name in served:
        print(f"        - {name}")

    print("\n  [2/3] checking the configured model ids are served ...")
    missing = sorted({ACTION_MODEL, CONSOLIDATOR_MODEL, JUDGE_MODEL} - set(served))
    if missing:
        print(f"        FAILED: not served here: {', '.join(missing)}")
        print("        Set AUTHMEM_ACTION_MODEL (and the consolidator/judge vars) to an id above.")
        return 1
    print("        ok")

    print("\n  [3/3] sending the real Module B request on both tool surfaces ...")
    if not all(probe(client, ACTION_MODEL, surface) for surface in ("direct", "gateway")):
        return 1

    print("\n  endpoint is fit to run against.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
