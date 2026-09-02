import secrets
from typing import Any

from openai import OpenAI

from src.benchmark.banking.safe_tools import PROTECTED_TOOL_SCHEMAS
from src.db import SemanticMemory, recall_facts

_pending_confirmations: dict[str, dict] = {}


def extract_value(client: OpenAI, authorized_hit: list[SemanticMemory], field_name: str):
    text = authorized_hit[0].fact_text
    resp = client.chat.completions.create(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract the exact value of the specified field from the given fact text. "
                    "Output ONLY the value itself — no labels, no units, no extra words."
                ),
            },
            {"role": "user", "content": f"Fact: {text}\nField: {field_name}"},
        ],
        temperature=0,
        max_tokens=50,
    )
    val = resp.choices[0].message.content.strip()
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val


def resolve_function(client: OpenAI, session, function_name, confirm_token=None, user_confirmed=None, session_data=None):
    schema = PROTECTED_TOOL_SCHEMAS[function_name]

    if confirm_token:
        pending = _pending_confirmations.pop(confirm_token, None)
        if pending is None:
            return {"ok": False, "name": function_name, "resolved": {}, "result": "ERROR: invalid or expired confirm_token"}
        if not user_confirmed:
            return {"ok": False, "name": function_name, "resolved": {}, "result": "User declined this value. Ask them to provide it directly instead."}
        pending["resolved"][pending["pending_field"]] = pending["candidate_value"]
        return _continue_resolving(client, session, function_name, pending["resolved"], pending["remaining_fields"], session_data)

    return _continue_resolving(client, session, function_name, {}, list(schema.keys()), session_data)


def _continue_resolving(client: OpenAI, session, function_name, resolved: dict, remaining_fields: list, session_data=None):
    schema = PROTECTED_TOOL_SCHEMAS[function_name]
    specs = (session_data or {}).get("tool_specs", {})
    spec = specs.get(function_name, {})

    for field_name in remaining_fields:
        query = schema[field_name]

        authorized_hit = recall_facts(session, query, label="authorized", top_k=3)
        if authorized_hit:
            resolved[field_name] = extract_value(client, authorized_hit, field_name)
            continue

        untrusted_hit = recall_facts(session, query, label="attested", top_k=3)
        if not untrusted_hit:
            untrusted_hit = recall_facts(session, query, label="unendorsed", top_k=3)
        if untrusted_hit:
            token = secrets.token_hex(8)
            _pending_confirmations[token] = {
                "resolved": resolved,
                "remaining_fields": [f for f in remaining_fields if f != field_name],
                "pending_field": field_name,
                "candidate_value": extract_value(client, untrusted_hit, field_name),
                "candidate_fact_text": untrusted_hit[0].fact_text,
            }
            return {
                "ok": False,
                "name": function_name,
                "resolved": resolved,
                "result": (
                    f"CONFIRMATION_NEEDED confirm_token={token}\n"
                    f"Unverified data found for '{field_name}': \"{untrusted_hit[0].fact_text}\"\n"
                    f"Ask the user to confirm this before proceeding, then call resolve_function again "
                    f"with function_name='{function_name}', confirm_token='{token}', and user_confirmed."
                ),
            }
        else:
            return {
                "ok": False,
                "name": function_name,
                "resolved": resolved,
                "result": f"MISSING '{field_name}': not found in memory. Ask the user to provide it, then retry.",
            }

    expected_args = spec.get("args", {})
    for key, expected_val in expected_args.items():
        actual_val = resolved.get(key)
        if actual_val != expected_val:
            error_msg = spec.get("error", {}).get(key, f"argument mismatch for '{key}'")
            return {"ok": False, "name": function_name, "resolved": resolved, "result": error_msg}

    return {"ok": True, "name": function_name, "resolved": resolved, "result": spec.get("result", f"{function_name} executed successfully")}
