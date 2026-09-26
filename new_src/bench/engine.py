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

import glob
import json
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from new_src.config import (
    API_KEY,
    API_KEYS,
    BASE_URL,
    DUTY_REST,
    DUTY_REST_MAX,
    DUTY_WORK,
    MAX_RETRIES,
    JUDGE_API_KEYS,
    JUDGE_BASE_URL,
    JUDGE_MIN_REQUEST_INTERVAL,
    JUDGE_PROXY,
    CONSOLIDATOR_API_KEYS,
    CONSOLIDATOR_BASE_URL,
    CONSOLIDATOR_MIN_REQUEST_INTERVAL,
    CONSOLIDATOR_PROXY,
    MIN_REQUEST_INTERVAL,
    REQUEST_TIMEOUT,
    TEMP_HIGH,
    TEMP_LOW,
)


class _KeyPool:
    """One OpenAI client per key for one endpoint, with that endpoint's own rotation state.

    There are two: the MAIN pool (the action agent, the consolidator, the write-path
    classifiers — AUTHMEM_BASE_URL / AUTHMEM_API_KEYS) and an optional JUDGE pool (Module A's
    judge and Module C's reference labeler — AUTHMEM_JUDGE_*). They are kept apart because they
    are usually different services: the agent on a local Ollama with no proxy, the judge on a
    hosted API that is only reachable through one. A request is always issued through the pool
    of the client it was given, so a caller holding the judge client can never end up talking to
    the agent's endpoint, or the other way round.
    """

    def __init__(self, base_url: str, keys: list[str], interval: float, proxy: str | None = None):
        self.base_url, self.keys, self.interval, self.proxy = base_url, keys, interval, proxy
        self.clients: list[OpenAI] = []
        self.dead: set[int] = set()
        self.cursor = 0
        self.last_request: dict[int, float] = {}

    def build(self) -> list[OpenAI]:
        with _POOL_LOCK:
            if not self.clients:
                http_client = None
                if self.proxy:
                    import httpx

                    http_client = httpx.Client(proxy=self.proxy, timeout=REQUEST_TIMEOUT)
                self.clients.extend(
                    OpenAI(base_url=self.base_url, api_key=key, timeout=REQUEST_TIMEOUT,
                           max_retries=MAX_RETRIES, http_client=http_client)
                    for key in self.keys
                )
            return self.clients


_POOL_LOCK = threading.Lock()
_MAIN = _KeyPool(BASE_URL, API_KEYS, MIN_REQUEST_INTERVAL)
_JUDGE = _KeyPool(JUDGE_BASE_URL, JUDGE_API_KEYS, JUDGE_MIN_REQUEST_INTERVAL, JUDGE_PROXY) \
    if JUDGE_BASE_URL else None
_CONSOLIDATOR = _KeyPool(CONSOLIDATOR_BASE_URL, CONSOLIDATOR_API_KEYS, CONSOLIDATOR_MIN_REQUEST_INTERVAL,
                         CONSOLIDATOR_PROXY) if CONSOLIDATOR_BASE_URL else None


def _pool() -> list[OpenAI]:
    return _MAIN.build()


def make_client() -> OpenAI:
    """The client callers hold. Requests are actually issued through the pool.

    Every caller in the benchmark obtains its client here, so handing back the pool's first
    member and routing requests over the whole pool is invisible to them. Keeping the parameter
    rather than reading a global is deliberate: the call sites stay explicit about which
    endpoint they are talking to.
    """
    return _MAIN.build()[0]


def make_judge_client() -> OpenAI:
    """The judge's client: its own endpoint when AUTHMEM_JUDGE_BASE_URL is set, otherwise the
    main one (the judge then runs wherever the agent runs, as before)."""
    return _JUDGE.build()[0] if _JUDGE is not None else make_client()


def make_consolidator_client() -> OpenAI:
    """The consolidator's client: its own endpoint when AUTHMEM_CONSOLIDATOR_BASE_URL is set,
    otherwise the main one, exactly as before."""
    return _CONSOLIDATOR.build()[0] if _CONSOLIDATOR is not None else make_client()


def _pool_of(client: OpenAI) -> _KeyPool:
    if _JUDGE is not None and client in _JUDGE.build():
        return _JUDGE
    if _CONSOLIDATOR is not None and client in _CONSOLIDATOR.build():
        return _CONSOLIDATOR
    return _MAIN


#: Errors that mean "not now" rather than "not ever". A hosted endpoint answers 429 when the
#: key is over its per-minute quota and 503 when the model is busy; both are cured by waiting.
#: A 400 or a 404 is not on this list on purpose — waiting will not fix a malformed request or
#: a model that does not exist, and retrying one hides the bug behind a delay.
_TRANSIENT = (RateLimitError, InternalServerError, APITimeoutError, APIConnectionError)

#: A key that is refused rather than throttled: revoked, unauthorized, or a project Google has
#: denied. Waiting does not help and every request dealt to it would fail, so it leaves the
#: rotation for the rest of the process instead of costing an episode each time it comes round.
_REFUSED = (AuthenticationError, PermissionDeniedError)

def _throttle(pool: _KeyPool, index: int) -> None:
    """Space this key's requests out by MIN_REQUEST_INTERVAL seconds.

    Reactive back-off alone is the wrong shape for a per-minute quota: it learns the limit by
    exceeding it, every single minute. Spacing the requests keeps the run under the limit
    instead, so the retries below stay a safety net rather than the normal path.

    The interval is per key, not global, because the quota is: two keys spaced four seconds
    apart each is eight requests a minute more than one key can send, not the same eight.
    """
    if pool.interval <= 0:
        return
    with _POOL_LOCK:
        wait = pool.last_request.get(index, 0.0) + pool.interval - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    with _POOL_LOCK:
        pool.last_request[index] = time.monotonic()


#: When the current stretch of work began. Reset by every pause.
_WORKING_SINCE = [time.monotonic()]


def _temperature() -> float | None:
    """The hottest thermal zone in degrees, or None if the machine exposes none.

    Read from sysfs rather than through lm-sensors: it is always present on Linux and needs no
    package installed. A machine without readable sensors is not an error — the schedule below
    still gives it its rest, and that is the half that must never depend on a sensor.
    """
    hottest = None
    for path in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            with open(path, encoding="utf-8") as handle:
                value = int(handle.read().strip()) / 1000.0
        except (OSError, ValueError):
            continue
        # Some zones report nonsense (0, or a raw millivolt reading). Ignore the implausible
        # rather than let one bad zone either mask real heat or force a permanent pause.
        if 20.0 < value < 130.0 and (hottest is None or value > hottest):
            hottest = value
    return hottest


def _rest(reason: str) -> None:
    """Stop issuing requests for a while so the machine can cool.

    Placed between requests, never during one: a pause inside an in-flight call would race the
    request timeout and be recorded as a transient failure. Waiting to start the next one costs
    nothing but wall-clock time, and the run resumes exactly where it was.
    """
    waited = 0.0
    step = 15.0
    print(f"\n  [pause] {reason}; resting {DUTY_REST / 60:.0f} min", flush=True)
    while waited < DUTY_REST_MAX:
        time.sleep(step)
        waited += step
        if waited < DUTY_REST:
            continue
        temperature = _temperature()
        if temperature is None or temperature <= TEMP_LOW:
            break
        # The scheduled rest is over but it is still hot: keep waiting rather than walking
        # straight back into the condition that caused the pause.
        if waited % 60 < step:
            print(f"  [pause] still {temperature:.0f} degrees, waiting for {TEMP_LOW:.0f}",
                  flush=True)
    temperature = _temperature()
    print(f"  [resume] after {waited / 60:.0f} min"
          + (f", {temperature:.0f} degrees" if temperature is not None else ""), flush=True)
    _WORKING_SINCE[0] = time.monotonic()


def _duty_cycle() -> None:
    """Rest if this stretch of work has run long enough, or if the machine is already too hot."""
    if DUTY_WORK <= 0:
        return
    worked = time.monotonic() - _WORKING_SINCE[0]
    temperature = _temperature()
    if temperature is not None and temperature >= TEMP_HIGH:
        _rest(f"{temperature:.0f} degrees, over the {TEMP_HIGH:.0f} limit")
    elif worked >= DUTY_WORK:
        _rest(f"{worked / 60:.0f} min of work")


def _next_index(pool: _KeyPool) -> int:
    """The next key in rotation, skipping any that has been refused outright."""
    with _POOL_LOCK:
        size = len(pool.clients) or 1
        live = [i for i in range(size) if i not in pool.dead]
        if not live:
            raise RuntimeError(
                f"every configured API key for {pool.base_url} was refused (401/403)."
            )
        index = live[pool.cursor % len(live)]
        pool.cursor += 1
    return index


def _request(make_call: Callable[[OpenAI], object], attempts: int = 5, key_pool: _KeyPool | None = None):
    """One request, waiting out transient refusals before giving up.

    Each attempt is dealt to the next key in turn, so a retry after a 429 lands on a different
    quota rather than on the one that just refused. The client already retries internally; this
    is the outer layer that survives the case where even those are exhausted. It matters
    because of what happens downstream: an exception here fails the episode, and five failed
    episodes in a row stop the phase. A rate limit must never be able to end a run.
    """
    key_pool = key_pool or _MAIN
    pool = key_pool.build()
    for attempt in range(attempts):
        _duty_cycle()
        index = _next_index(key_pool)
        _throttle(key_pool, index)
        try:
            return make_call(pool[index])
        except _REFUSED as error:
            with _POOL_LOCK:
                first = index not in key_pool.dead
                key_pool.dead.add(index)
            if first:
                print(f"    [key {index + 1}/{len(pool)} refused: {type(error).__name__}] "
                      f"dropping it from the rotation for this run", flush=True)
            if len(key_pool.dead) >= len(pool):
                raise
            continue
        except _TRANSIENT as error:
            if attempt == attempts - 1:
                raise
            delay = min(5.0 * 2 ** attempt, 60.0) + random.uniform(0, 2.0)
            print(f"    [waiting {delay:.0f}s] {type(error).__name__} on key {index + 1}"
                  f"/{len(pool)} (attempt {attempt + 1}/{attempts - 1})", flush=True)
            time.sleep(delay)
    raise RuntimeError("unreachable")


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

#: Ollama accepts `chat_template_kwargs` without error and ignores it: Qwen3.5 on Ollama went on
#: thinking through every classifier call, about 80 s per pilot case against 2 s with this. Its
#: OpenAI-compatible switch is `reasoning_effort: none`, which it also accepts for models that do
#: not reason.
_NO_THINK_BODY_OLLAMA = {"reasoning_effort": "none"}


def _no_think_body(base: str) -> dict:
    return _NO_THINK_BODY_OLLAMA if ":11434" in base else _NO_THINK_BODY


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
    key_pool = _pool_of(client)
    if thinking or _NO_THINK_SUPPORTED.get(base) is False:
        return _request(lambda c: c.chat.completions.create(**kwargs), key_pool=key_pool)
    try:
        response = _request(lambda c: c.chat.completions.create(**kwargs,
                                                                    extra_body=_no_think_body(base)),
                            key_pool=key_pool)
    except _TRANSIENT:
        # A rate limit says nothing about whether the server understands the option. Recording
        # it as "unsupported" would silently turn thinking back on for the rest of the run.
        raise
    except Exception:  # noqa: BLE001 - any refusal of the option, not just BadRequest
        _NO_THINK_SUPPORTED[base] = False
        return _request(lambda c: c.chat.completions.create(**kwargs), key_pool=key_pool)
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
