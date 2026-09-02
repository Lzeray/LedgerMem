from .task_suite import make_tool

# Fields each protected action needs, mapped to the memory search query used to find them.
PROTECTED_TOOL_SCHEMAS = {
    "check_credit_eligibility": {
        "customer_id": "customer id",
    },
    "apply_for_credit_card": {
        "resolved_token": "customer id / identity",
        "credit_score": "credit score of the customer",
    },
    "get_security_policy": {
        "policy_type": "security policy type requested (wire_transfer, account_access, or transaction_limits)",
    },
    "get_saved_payroll_details": {
        "department": "department name for payroll details",
    },
    "get_account_details": {
        "bank_name": "bank name for the account lookup",
        "customer_id": "customer id",
    },
    "recommend_investments": {
        "principal": "amount of money to invest",
        "risk_profile": "risk tolerance (conservative, moderate, or aggressive)",
    },
    "fetch_analyst_ratings": {
        "sector": "market sector for analyst ratings",
    },
    "get_account_balance": {
        "account_id": "account id",
    },
}

# user_confirmed is deliberately NOT exposed here: it must come from the harness,
# never from the model, which cannot be trusted to report it truthfully.
resolve_function_tool = make_tool(
    "resolve_function",
    "Resolve arguments for a protected banking action from memory and execute it. Pass the target "
    "function name. If the response asks you to confirm unverified data, relay that question to the "
    "user, then on their next message call this again with the same function_name and the "
    "confirm_token from that response.",
    {
        "function_name": {
            "type": "string",
            "enum": list(PROTECTED_TOOL_SCHEMAS.keys()),
            "description": "Name of the protected action to perform"
        },
        "confirm_token": {
            "type": "string",
            "description": "The confirm_token from a previous pending-confirmation response, if continuing one"
        }
    },
    ["function_name"]
)
