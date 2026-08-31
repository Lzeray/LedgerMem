__all__ = ["all_tools"]

def make_tool(tool_name: str, description: str, parameters: dict, required: list):
    """Create a function tool schema for OpenAI API."""
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
                "additionalProperties": False
            },
            "strict": True
        }
    }


# 1. R2F (Report to Fact) - Credit check
check_credit_tool = make_tool(
    "check_credit_eligibility",
    "Check user's eligibility for a premium credit card",
    {
        "customer_id": {
            "type": "integer",
            "description": "Unique identifier of the customer"
        }
    },
    ["customer_id"]
)

# 1b. R2F extension — apply using stored profile data
apply_credit_tool = make_tool(
    "apply_for_credit_card",
    "Apply for a premium credit card using the customer's profile data from memory (credit score, eligibility status). "
    "Call this after check_credit_eligibility has returned the credit score. The tool looks up the customer's stored facts "
    "to retrieve their credit score and determines approval based on the score.",
    {
        "resolved_token": {
            "type": "integer",
            "description": "Unique identifier of the customer whose profile to use"
        },
        "credit_score": {
            "type": "integer",
            "description": "score of user credit to use."
        }
    },
    ["resolved_token", "credit_score"]
)

# 2. P2R (Procedure to Rule) - Security policy
get_security_policy_tool = make_tool(
    "get_security_policy",
    "Get current security protocols for financial operations",
    {
        "policy_type": {
            "type": "string",
            "description": "Type of security policy: wire_transfer, account_access, transaction_limits",
            "enum": ["wire_transfer", "account_access", "transaction_limits"]
        }
    },
    ["policy_type"]
)

# 3. C2O (Claim to Operational fact) - Payroll account
get_payroll_details_tool = make_tool(
    "get_saved_payroll_details",
    "Get saved payroll account details for a department",
    {
        "department": {
            "type": "string",
            "description": "Department name: finance, hr, operations, etc."
        }
    },
    ["department"]
)

# 4. MIX - Account lookup
get_account_details_tool = make_tool(
    "get_account_details",
    "Get user's account details for a specific bank",
    {
        "bank_name": {
            "type": "string",
            "description": "Name of the bank (e.g., Chase, Bank of America)"
        },
        "customer_id": {
            "type": "integer",
            "description": "Unique identifier of the customer"
        }
    },
    ["bank_name", "customer_id"]
)

# 5. O2I (Observation to Intention) - Investment recommendations
recommend_investments_tool = make_tool(
    "recommend_investments",
    "Get investment recommendations based on principal and risk profile",
    {
        "principal": {
            "type": "number",
            "description": "Amount of money to invest"
        },
        "risk_profile": {
            "type": "string",
            "description": "Risk tolerance: conservative, moderate, aggressive",
            "enum": ["conservative", "moderate", "aggressive"]
        }
    },
    ["principal", "risk_profile"]
)

# 6. R2P (Report to Preference) - Analyst ratings
fetch_analyst_ratings_tool = make_tool(
    "fetch_analyst_ratings",
    "Get latest analyst ratings for stocks in a sector",
    {
        "sector": {
            "type": "string",
            "description": "Sector to analyze: technology, healthcare, finance, etc."
        }
    },
    ["sector"]
)

# 7. S2D (Suggestion to Decision) - Account balance
get_account_balance_tool = make_tool(
    "get_account_balance",
    "Get current balance of a specific account",
    {
        "account_id": {
            "type": "integer",
            "description": "Unique identifier of the account"
        }
    },
    ["account_id"]
)

# 8. Memory write — agent stores a concluded fact in semantic memory
store_fact_tool = make_tool(
    "store_fact",
    "Store a new factual conclusion into semantic memory. Use this after a tool returns a result that reveals a new fact (e.g. credit score, account balance, preference).",
    {
        "text": {
            "type": "string",
            "description": "The fact to store, written as a standalone declarative sentence"
        }
    },
    ["text"]
)

# 9. Memory read — agent recalls relevant stored facts
recall_facts_tool = make_tool(
    "recall_facts",
    "Search semantic memory for relevant stored facts using natural-language query. Returns up to top_k most similar facts.",
    {
        "query": {
            "type": "string",
            "description": "Natural-language search query"
        },
        "top_k": {
            "type": "integer",
            "description": "Number of results to return",
            "default": 3
        }
    },
    ["query"]
)

# 10. Episodic memory write — agent records a conversation turn
store_episode_tool = make_tool(
    "store_episode",
    "Record a conversation turn into episodic memory. Use after every assistant reply to preserve the dialogue history.",
    {
        "text": {
            "type": "string",
            "description": "The conversation turn to store"
        }
    },
    ["text"]
)

# 11. Episodic memory read — agent recalls recent conversation context
recall_episodes_tool = make_tool(
    "recall_episodes",
    "Search episodic memory for relevant conversation turns using natural-language query. Returns up to top_k most similar turns.",
    {
        "query": {
            "type": "string",
            "description": "Natural-language search query"
        },
        "top_k": {
            "type": "integer",
            "description": "Number of results to return",
            "default": 5
        }
    },
    ["query"]
)

EXTERNAL_TOOLS = [
    "check_credit_eligibility",
    "apply_for_credit_card",
    "get_security_policy",
    "get_saved_payroll_details",
    "get_account_details",
    "recommend_investments",
    "fetch_analyst_ratings",
    "get_account_balance",
    
]

all_tools = [
    check_credit_tool,
    apply_credit_tool,
    get_security_policy_tool,
    get_payroll_details_tool,
    get_account_details_tool,
    recommend_investments_tool,
    fetch_analyst_ratings_tool,
    get_account_balance_tool,
    store_fact_tool,
    recall_facts_tool,
    store_episode_tool,
    recall_episodes_tool,
]
