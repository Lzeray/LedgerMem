"""
Talking to the model. Nothing here knows about banking or authority.

Two entry points:

  complete()        one chat completion, retried at escalating temperature if the model
                    returns a fully empty response (no content, no tool call). That is a
                    real, reproducible failure of small local models on certain phrasings,
                    not a hypothetical, and a little temperature reliably breaks it.

  run_tool_turn()   one agent turn: send the messages, dispatch whatever tool calls come
                    back through a caller-supplied executor, and feed the results back until
                    the model stops calling tools or the call budget runs out. Returns every
                    call it made, which is what the action predicate is evaluated against.

System prompts written against these helpers must avoid literal code/pseudo-code syntax:
qwen-family local models reliably deadlock into an empty response when a prompt contains
something like `some_function(arg='x')`. Describe tools in prose.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from openai import OpenAI

from new_src.config import API_KEY, BASE_URL


def make_client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


@dataclass
class MadeCall:
    name: str
    arguments: dict
    raw_arguments: str
    malformed: bool = False
    result: object = None


@dataclass
class TurnResult:
    calls: list[MadeCall] = field(default_factory=list)
    content: str = ""
    budget_exhausted: bool = False


#: Hard ceiling for the automatic budget escalation below.
TOKEN_CEILING = 6000

#: Servers that accepted (or refused) a request to switch a reasoning model's thinking off,
#: keyed by base URL. Probed once per endpoint rather than on every call: a server that does
#: not understand the option answers 400, and asking it again for every classification would
#: double the traffic for nothing.
_NO_THINK_SUPPORTED: dict[str, bool] = {}

#: How vLLM and Ray Serve expose Qwen's thinking switch. Sent only when `thinking=False`.
_NO_THINK_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


def _create(client: OpenAI, kwargs: dict, thinking: bool):
    """One request, optionally asking a reasoning model not to think.

    Thinking is right for the AGENT under test — that is the system being measured. It is pure
    cost for the harness's own classifiers, which answer with a single word and whose
    deliberation nobody reads: on Qwen3.5 a yes/no answer costs around 220 reasoning tokens, and
    Module C makes eight to twelve such calls per episode.

    Turning it off is a methodological choice, not a free optimisation: the harness's
    classifier becomes a weaker classifier, and that has to be stated where the numbers are
    reported. It is applied only to the harness's own auxiliary calls, never to the action
    agent.
    """
    base = str(getattr(client, "base_url", ""))
    if thinking or _NO_THINK_SUPPORTED.get(base) is False:
        return client.chat.completions.create(**kwargs)
    try:
        response = client.chat.completions.create(**kwargs, extra_body=_NO_THINK_BODY)
    except Exception:  # noqa: BLE001 - any refusal of the option, not just BadRequest
        _NO_THINK_SUPPORTED[base] = False
        return client.chat.completions.create(**kwargs)
    _NO_THINK_SUPPORTED[base] = True
    return response


def complete(
    client: OpenAI,
    model: str,
    messages: list,
    tools: list | None = None,
    temperatures: tuple[float, ...] = (0.0, 0.3, 0.6, 0.9),
    max_tokens: int = 2000,
    thinking: bool = True,
):
    """One completion, retried when the model returns nothing usable.

    There are two distinct reasons for an empty response and they need different remedies.

    A small local model sometimes deadlocks on a phrasing and returns nothing at any budget;
    a little temperature reliably breaks that, which is what the temperature ladder is for.

    A REASONING model returns nothing for a completely different reason: it spends the token
    budget thinking and never reaches its answer. The response then carries
    `finish_reason == "length"` with empty content, and no amount of temperature helps — it
    needs room. Qwen3.5 served through Ray Serve spends around 220 tokens of reasoning before
    answering a yes/no question, so every short-answer call in this benchmark (the classifier
    at 40 tokens, the role predictor and the judge at 8) came back empty, the classifier
    returned its fail-closed value for every record, and the gate refused H+ as well as H-.
    That reads as a defense with zero utility and is entirely an artifact of the budget.

    `max_tokens` is a ceiling, not a cost: a model that answers in five tokens is charged for
    five however high the cap. So the defaults here are generous and the escalation below is
    free on models that do not think.
    """
    message = None
    budget = max_tokens
    for temperature in temperatures:
        kwargs = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": budget}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        choice = _create(client, kwargs, thinking).choices[0]
        message = choice.message
        if message.content or getattr(message, "tool_calls", None):
            return message
        if getattr(choice, "finish_reason", None) == "length" and budget < TOKEN_CEILING:
            # Out of room rather than stuck: give it room instead of jiggling the temperature.
            budget = min(budget * 4, TOKEN_CEILING)
    return message


def complete_text(client: OpenAI, model: str, system: str, user: str, max_tokens: int = 4000,
                  thinking: bool = True) -> str:
    message = complete(
        client, model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tools=None, max_tokens=max_tokens, thinking=thinking,
    )
    return (message.content or "").strip()


def run_tool_turn(
    client: OpenAI,
    model: str,
    messages: list,
    tools: list,
    execute: Callable[[str, dict], str],
    max_calls: int = 4,
    verbose: bool = True,
) -> TurnResult:
    result = TurnResult()
    used = 0

    while used < max_calls:
        message = complete(client, model, messages, tools)
        messages.append(message)
        if message.content and verbose:
            print(f"    [assistant] {message.content.strip()[:400]}")
        result.content = message.content or result.content

        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            break

        for call in tool_calls:
            raw = call.function.arguments or ""
            try:
                arguments = json.loads(raw) if raw.strip() else {}
                malformed = not isinstance(arguments, dict)
                if malformed:
                    arguments = {}
            except json.JSONDecodeError:
                arguments, malformed = {}, True

            made = MadeCall(name=call.function.name, arguments=arguments, raw_arguments=raw, malformed=malformed)
            if verbose:
                print(f"    [call] {made.name}({made.arguments})" + ("  <malformed arguments>" if malformed else ""))

            output = execute(made.name, made.arguments)
            made.result = output
            result.calls.append(made)
            if verbose:
                print(f"    [result] {str(output)[:300]}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(output)})
            used += 1

    result.budget_exhausted = used >= max_calls
    return result
