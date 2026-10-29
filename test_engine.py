from engine import evaluate, verify_draft


def test_flu_like_pattern():
    result = evaluate(["fever", "cough", "fatigue", "body_aches"])
    assert result["urgency"] == "NON_EMERGENCY_SCREENING"
    assert any(h["id"] == "influenza_like_pattern" for h in result["hypotheses"])


def test_emergency_override():
    result = evaluate(["chest_pain", "severe_shortness_of_breath"])
    assert result["urgency"] == "EMERGENCY"
    assert len(result["red_flags"]) >= 2


def test_unsupported_absolute_claim_rejected():
    result = evaluate(["fever", "cough"])
    verification = verify_draft("You definitely have pneumonia. Take an antibiotic.", result)
    assert verification["verified"] is False
    assert len(verification["rejected_claims"]) >= 1


def test_safe_disclaimer_not_false_positive():
    result = evaluate(["runny_nose"])
    verification = verify_draft(
        "This is not a diagnosis. The symptoms are compatible with more than one pattern.",
        result,
    )
    assert verification["verified"] is True


def test_emergency_dismissal_rejected():
    result = evaluate(["chest_pain", "severe_shortness_of_breath"])
    verification = verify_draft(
        "This is probably just flu. You can manage it at home.",
        result,
    )
    assert verification["verified"] is False
    assert verification["emergency_override"] is True


def test_supported_pattern_accepted():
    result = evaluate(["runny_nose", "sneezing", "itchy_eyes"])
    verification = verify_draft(
        "The symptoms are compatible with an allergic-rhinitis-like pattern. This is not a diagnosis.",
        result,
    )
    assert verification["verified"] is True


if __name__ == "__main__":
    test_flu_like_pattern()
    test_emergency_override()
    test_unsupported_absolute_claim_rejected()
    test_safe_disclaimer_not_false_positive()
    test_emergency_dismissal_rejected()
    test_supported_pattern_accepted()
    print("All tests passed.")
