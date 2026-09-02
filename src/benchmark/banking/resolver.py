from openai import OpenAI

from src.benchmark.banking.safe_tools import PROTECTED_TOOL_SCHEMAS
from src.db import SemanticMemory, recall_facts


def extract_value(client: OpenAI, hit: list[SemanticMemory], function_name, field_name: str):
    search = PROTECTED_TOOL_SCHEMAS[function_name][field_name]
    text = "\n".join([memo.fact_text for memo in hit])
    resp = client.chat.completions.create(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "system",
                "content": (
                    f"Extract the exact value of the specified field from the given fact text. Searched by: {field_name} = {search}"
                    "Output ONLY the value itself — no labels, no units, no extra words. If no searching fact, return 'NO'"
                ),
            },
            {"role": "user", "content": f"Facts:\n{text}\nField: {field_name}"},
        ],
        temperature=0,
        max_tokens=50,
    )
    val = resp.choices[0].message.content.strip()
    print(f"\nSearch for {field_name}, found: {val}, memory: \n", "\n ".join(memo.fact_text for memo in hit))
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val


def resolve_function(client: OpenAI, session, function_name, session_data=None):
    schema = PROTECTED_TOOL_SCHEMAS[function_name]
    resolved = {}
    to_confirm = []
    to_query = []

    for field_name in schema:
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
                to_confirm.append({
                    "field_name": field_name,
                    "value": value,
                    "label": "attested",
                })
                continue
            
        untrusted_hit = recall_facts(session, query, label="unendorsed", top_k=3)
        if untrusted_hit:
            value = extract_value(client, untrusted_hit, function_name, field_name)
            if value != "NO":
                to_confirm.append({
                    "field_name": field_name,
                    "value": value,
                    "label": "attested",
                    "query": f"{field_name}={value}"
                })
                continue
                    
        to_query.append(field_name)
            
    print("\nresolved: ", resolved)
    print("to confirm: ", to_confirm)
    print("to query: ", ", ".join(to_query))
    
    output = ""
    if to_query:
        output += f"Please, give some necessary data: {", ".join([query for query in to_query])}.\n"

    if to_confirm:
        output += f"Do you confirm this fields: {", ".join([confirm["query"] for confirm in to_confirm])}.\n"
        
    if output:
        return {"ok": True, "name": function_name, "resolved": resolved, "result": output}
    
    specs = (session_data or {}).get("tool_specs", {})
    spec = specs.get(function_name, {})
        
    expected_args = spec.get("args", {})
    for key, expected_val in expected_args.items():
        actual_val = resolved.get(key)
        if actual_val != expected_val:
            error_msg = spec.get("error", {}).get(key, f"argument mismatch for '{key}'")
            return {"ok": False, "name": function_name, "resolved": resolved, "result": error_msg}

    return {"ok": True, "name": function_name, "resolved": resolved, "result": spec.get("result", f"{function_name} executed successfully")}
