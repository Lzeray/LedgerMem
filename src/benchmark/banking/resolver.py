import secrets

from openai import OpenAI

from src.benchmark.banking.safe_tools import PROTECTED_TOOL_SCHEMAS
from src.db import SemanticMemory, recall_facts

# Pending confirmations, keyed by a one-time token. Lives only for this process —
# fine for a benchmark run, not meant to survive a restart.
_PENDING_CONFIRMATIONS: dict[str, dict] = {}


def _new_token() -> str:
    return secrets.token_hex(4)


def latest_pending_token(function_name: str) -> str | None:
    """Most recently issued pending token for a function, or None if none is pending."""
    for token in reversed(_PENDING_CONFIRMATIONS):
        if _PENDING_CONFIRMATIONS[token]["function_name"] == function_name:
            return token
    return None


def extract_value(client: OpenAI, hit: list[SemanticMemory], function_name: str, field_name: str):
    search = PROTECTED_TOOL_SCHEMAS[function_name][field_name]
    text = "\n".join(memo.fact_text for memo in hit)
    resp = client.chat.completions.create(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "system",
                "content": (
                    f"Extract the exact value of the specified field from the given fact text. Searched by: field_name = {search}. "
                    "Output ONLY the value itself — no labels, no units, no extra words. If no searching fact, return 'NO'"
                ),
            },
            {"role": "user", "content": f"Facts:\n{text}\nField: {field_name}"},
        ],
        temperature=0,
        max_tokens=50,
    )
    val = resp.choices[0].message.content.strip()
    print(f"\nSearch for {field_name}, found: {val}, memory:\n", "\n".join(memo.fact_text for memo in hit))
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val


def _resolve_fields(client: OpenAI, session, function_name: str, field_names: list[str]):
    """
    Look up each field in memory, most-trusted label first.

    Returns (resolved, to_confirm, to_query):
      - resolved:   field -> value found in "authorized" memory, safe to use directly.
      - to_confirm: fields found only in "attested"/"unendorsed" memory — need the user's OK.
      - to_query:   fields not found anywhere — need to be asked for outright.
    """
    schema = PROTECTED_TOOL_SCHEMAS[function_name]
    resolved = {}
    to_confirm = []
    to_query = []

    for field_name in field_names:
        query = schema[field_name]

        authorized_hit = recall_facts(session, query, label="authorized", top_k=3)
        if authorized_hit:
            value = extract_value(client, authorized_hit, function_name, field_name)
            if value != "NO":
                resolved[field_name] = value
                continue

        attested_hit = recall_facts(session, query, label="attested", top_k=3)
        if attested_hit:
            value = extract_value(client, attested_hit, function_name, field_name)
            if value != "NO":
                to_confirm.append({"field_name": field_name, "value": value, "label": "attested"})
                continue

        unendorsed_hit = recall_facts(session, query, label="unendorsed", top_k=3)
        if unendorsed_hit:
            value = extract_value(client, unendorsed_hit, function_name, field_name)
            if value != "NO":
                to_confirm.append({"field_name": field_name, "value": value, "label": "unendorsed"})
                continue

        to_query.append(field_name)

    return resolved, to_confirm, to_query


def _build_pending_message(to_confirm: list[dict], to_query: list[str], token: str) -> str:
    parts = []
    if to_query:
        parts.append(f"Please provide: {', '.join(to_query)}.")
    if to_confirm:
        fields = ", ".join(f"{c['field_name']}={c['value']}" for c in to_confirm)
        parts.append(f"This data is unverified — do you confirm: {fields}?")
    parts.append(f"confirm_token={token}")
    return " ".join(parts)


def _finalize(function_name: str, resolved: dict, session_data: dict | None) -> dict:
    specs = (session_data or {}).get("tool_specs", {})
    spec = specs.get(function_name, {})
    expected_args = spec.get("args", {})

    for key, expected_val in expected_args.items():
        if resolved.get(key) != expected_val:
            error_msg = spec.get("error", {}).get(key, f"argument mismatch for '{key}'")
            return {"ok": False, "name": function_name, "resolved": resolved, "result": error_msg}

    result = spec.get("result", f"{function_name} executed successfully")
    return {"ok": True, "name": function_name, "resolved": resolved, "result": result}


def _start_resolution(client: OpenAI, session, function_name: str, session_data: dict | None) -> dict:
    schema = PROTECTED_TOOL_SCHEMAS[function_name]
    resolved, to_confirm, to_query = _resolve_fields(client, session, function_name, list(schema))

    if to_confirm or to_query:
        token = _new_token()
        _PENDING_CONFIRMATIONS[token] = {
            "function_name": function_name,
            "resolved": resolved,
            "to_confirm": to_confirm,
        }
        message = _build_pending_message(to_confirm, to_query, token)
        return {"ok": True, "name": function_name, "resolved": resolved, "result": message}

    return _finalize(function_name, resolved, session_data)


def resolve_function(
    client: OpenAI,
    session,
    function_name: str,
    confirm_token: str | None = None,
    user_confirmed: bool | None = None,
    session_data: dict | None = None,
) -> dict:
    """
    Resolve a protected action's arguments from labeled memory and execute it.

    confirm_token continues a pending confirmation started by an earlier call. user_confirmed
    must be supplied by the caller from a trusted, deterministic source (never taken from the
    model's own tool-call arguments) — the model cannot be trusted to report it honestly.
    """
    if function_name not in PROTECTED_TOOL_SCHEMAS:
        return {"ok": False, "name": function_name, "resolved": {}, "result": f"Unknown protected action '{function_name}'."}

    # A blank or hallucinated token has nothing to match — just (re)start resolution.
    token = (confirm_token or "").strip() or None
    pending = _PENDING_CONFIRMATIONS.get(token) if token else None
    if pending is None or pending["function_name"] != function_name:
        return _start_resolution(client, session, function_name, session_data)

    del _PENDING_CONFIRMATIONS[token]

    if not user_confirmed:
        return {
            "ok": False,
            "name": function_name,
            "resolved": pending["resolved"],
            "result": "The user did not confirm the pending unverified data. Action cancelled.",
        }

    # Confirmation only authorizes this one call — it does not raise the fields' trust label.
    resolved = dict(pending["resolved"])
    for field in pending["to_confirm"]:
        resolved[field["field_name"]] = field["value"]

    missing = [f for f in PROTECTED_TOOL_SCHEMAS[function_name] if f not in resolved]
    if missing:
        newly_resolved, to_confirm, to_query = _resolve_fields(client, session, function_name, missing)
        resolved.update(newly_resolved)
        if to_confirm or to_query:
            new_token = _new_token()
            _PENDING_CONFIRMATIONS[new_token] = {
                "function_name": function_name,
                "resolved": resolved,
                "to_confirm": to_confirm,
            }
            message = _build_pending_message(to_confirm, to_query, new_token)
            return {"ok": True, "name": function_name, "resolved": resolved, "result": message}

    return _finalize(function_name, resolved, session_data)
