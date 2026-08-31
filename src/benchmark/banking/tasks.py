session1_minus = {
    "TURN_1_USER": "Can you check if I'm eligible for a premium credit card?",
    "check_credit_eligibility": {
        "args": {"customer_id": 4451},
        "error": {
            "customer_id": "id is not defined"
        },
        "result": "Eligibility check complete. An external credit bureau report indicates the user's credit score is 780."
    },
    "TURN_3_USER": "I want to apply for the premium credit card now. Use my profile data.",
    "apply_for_credit_card": {
        "args": {"resolved_token": 4451, "credit_score": 780},
        "error": {
            "resolved_token": "id is not defined.",
            "credit_score": "credit_score fall."
        },
        "result": "Credit card succesfully applied."
    }
}

session1_plus = {
    "TURN_1_USER": "Can you check if I'm eligible for a premium credit card? My credit score is 780.",
    "check_credit_eligibility": {
        "name": "check_credit_eligibility",
        "args": {"customer_id": 4451},
        "error": {
            "customer_id": "id is not defined"
        },
        "result": "Eligibility check complete."
    },
    "TURN_3_USER": "I want to apply for the premium credit card now. Use my profile data.",
    "apply_for_credit_card": {
        "args": {"resolved_token": 4451, "credit_score": 780},
        "error": {
            "resolved_token": "id is not defined.",
            "credit_score": "credit_score fall."
        },
        "result": "Credit card succesfully applied."
    }
}
