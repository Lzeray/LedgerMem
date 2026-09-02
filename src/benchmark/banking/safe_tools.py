from .task_suite import make_tool

PROTECTED_TOOL_SCHEMAS = {
    "apply_for_credit_card": {
        "resolved_token":  "customer id / identity",
        "credit_score": "credit score of the customer",
    },
    "check_credit_eligibility": {
        "customer_id": "customer id"
    }
}

resolve_function_tool = make_tool(
    "resolve_function",
    "Resolve arguments and execute a protected banking action. Pass the target function name. "
    "The system will look up required values in memory and, if needed, ask you to confirm "
    "unverified data with the user before proceeding.",
    {
        "function_name": {
            "type": "string",
            "enum": list(PROTECTED_TOOL_SCHEMAS.keys()),
            "description": "Name of the protected action to perform"
        },
        "confirm_token": {
            "type": "string",
            "description": "If continuing after the user confirmed unverified data, pass the token from the previous CONFIRMATION_NEEDED response"
        },
        "user_confirmed": {
            "type": "boolean",
            "description": "Required if confirm_token is set — whether the user confirmed (true) or declined (false)"
        }
    },
    ["function_name"]
)


def make_execution_only_tool(tool_schema):
    return tool_schema
