"""
Domain-agnostic agent turn-loop engine.

Nothing in this module knows about banking, memory labels, or any specific tool —
it only knows how to talk to an OpenAI-compatible chat-completions endpoint and
dispatch whatever tool calls come back, via callbacks the caller supplies. Any
benchmark (banking or otherwise) can drive its turns through this loop.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from openai import OpenAI

ExecuteTool = Callable[[str, dict], Any]
FormatOutput = Callable[[str, Any], str]


@dataclass
class ToolCallRecord:
    name: str
    args: dict
    result: Any


@dataclass
class TurnLoopResult:
    calls: list[ToolCallRecord] = field(default_factory=list)
    exceeded_max_calls: bool = False


def complete_with_retry(
    client: OpenAI,
    model: str,
    messages: list,
    tools: list,
    temperatures: tuple[float, ...] = (0, 0.3, 0.6, 0.9),
    max_tokens: int = 300,
):
    """
    Call the model, retrying with escalating temperature if it returns a fully empty
    response (no content, no tool call). Small local models occasionally deadlock into
    this empty response for certain phrasings even at temperature 0 — reproduces across
    model sizes, not just the smallest one. A little temperature reliably breaks it.
    """
    msg = None
    for temperature in temperatures:
        msg = client.chat.completions.create(
            model=model, messages=messages, tools=tools, tool_choice="auto",
            temperature=temperature, max_tokens=max_tokens,
        ).choices[0].message
        if msg.content or msg.tool_calls:
            return msg
    return msg


def run_turn_loop(
    client: OpenAI,
    model: str,
    messages: list,
    tools: list,
    execute_tool: ExecuteTool,
    format_output: FormatOutput,
    max_tool_calls: int = 10,
) -> TurnLoopResult:
    """
    Run the request/response/tool-call loop until the model stops calling tools or
    max_tool_calls is hit. Mutates `messages` in place (appends the assistant and tool
    messages), matching what the OpenAI tool-calling protocol requires.

    execute_tool(name, args) -> result: runs one tool call and returns its result
    (any shape the caller wants — a plain string, or a richer dict).

    format_output(name, result) -> str: renders that result into the string that goes
    back to the model as the tool-role message content.
    """
    import json

    result = TurnLoopResult()
    seen_tool_keys = set()  # (name, sorted-args) already executed this loop — de-dupes exact repeats
    tool_call_count = 0

    while tool_call_count < max_tool_calls:
        msg = complete_with_retry(client, model, messages, tools)
        messages.append(msg)

        if msg.content:
            print(f"\n  [ASSISTANT] {msg.content}")

        if not msg.tool_calls:
            break

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            tool_key = (tc.function.name, tuple(sorted(args.items())))
            print(f"    [CALL] {tc.function.name}({args})")

            if tool_key in seen_tool_keys:
                output = "(same as previous call, skipping)"
                print(f"    [RESULT] {output}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
                tool_call_count += 1
                continue
            seen_tool_keys.add(tool_key)

            call_result = execute_tool(tc.function.name, args)
            output = format_output(tc.function.name, call_result)
            print(f"    [RESULT] {output}")

            result.calls.append(ToolCallRecord(name=tc.function.name, args=args, result=call_result))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
            tool_call_count += 1

    result.exceeded_max_calls = tool_call_count >= max_tool_calls
    return result
