import json
import re
from pathlib import Path
from typing import Dict, List, Any

KB_PATH = Path(__file__).resolve().parent / "data" / "knowledge_base.json"

with open(KB_PATH, "r", encoding="utf-8") as f:
    KB = json.load(f)


def normalize(symptoms: List[str]) -> set[str]:
    if symptoms is None:
        return set()
    if isinstance(symptoms, str):
        symptoms = [symptoms]
    return {str(s).strip().lower() for s in symptoms if s and str(s).strip()}


def _red_flag_by_id(rule_id: str) -> dict:
    return next((x for x in KB["red_flags"] if x["id"] == rule_id), None)


def _add_red_flag(red_flag_hits: list, rule_id: str):
    rule = _red_flag_by_id(rule_id)
    if rule and rule not in red_flag_hits:
        red_flag_hits.append(rule)




TEXT_SYMPTOM_ALIASES = {
    "fever": ["fever", "high temperature", "temperature"],
    "cough": ["cough", "coughing"],
    "sore_throat": ["sore throat", "throat pain", "painful throat"],
    "runny_nose": ["runny nose", "running nose", "nose is running"],
    "sneezing": ["sneezing", "sneeze"],
    "fatigue": ["fatigue", "very tired", "extreme tiredness", "tiredness"],
    "body_aches": ["body aches", "body pain", "muscle aches", "muscle pain"],
    "headache": ["headache", "head pain"],
    "nausea": ["nausea", "feeling sick"],
    "light_sensitivity": ["light sensitivity", "sensitive to light", "photophobia"],
    "vomiting": ["vomiting", "vomit", "throwing up"],
    "diarrhea": ["diarrhea", "loose stools", "loose motion", "loose motions"],
    "abdominal_pain": ["abdominal pain", "stomach pain", "belly pain"],
    "burning_urination": ["burning urination", "burning when urinating", "painful urination"],
    "frequent_urination": ["frequent urination", "urinating frequently", "pee frequently"],
    "lower_abdominal_pain": ["lower abdominal pain", "lower stomach pain"],
    "itchy_eyes": ["itchy eyes", "itching eyes", "watery itchy eyes"],
    "chest_pain": ["chest pain", "pain in my chest", "chest hurts"],
    "severe_shortness_of_breath": ["severe shortness of breath", "severe difficulty breathing", "can't breathe", "cannot breathe", "struggling to breathe", "difficulty breathing"],
    "one_sided_weakness": ["one-sided weakness", "one sided weakness", "weakness on one side"],
    "speech_difficulty": ["speech difficulty", "difficulty speaking", "slurred speech"],
    "facial_droop": ["facial droop", "face drooping", "drooping face"],
    "sudden_vision_change": ["sudden vision change", "sudden change in vision", "sudden blurred vision"],
    "sudden_loss_of_balance": ["sudden loss of balance", "suddenly losing balance", "loss of balance"],
    "sudden_severe_headache": ["sudden severe headache", "worst headache", "sudden terrible headache"],
    "confusion": ["confusion", "confused", "new confusion"],
    "fainting": ["fainting", "fainted", "passed out", "loss of consciousness"],
    "no_urination": ["no urination", "not urinating", "haven't urinated", "have not urinated"],
    "very_dark_urine": ["very dark urine", "dark urine"],
    "rapid_heartbeat": ["rapid heartbeat", "racing heart", "heart racing", "fast heartbeat"],
    "rapid_breathing": ["rapid breathing", "breathing very fast", "fast breathing"],
}


def extract_symptoms_from_text(text: str) -> List[str]:
    """Extract known symptom IDs from free-text notes using transparent aliases.

    This is a convenience layer, not a diagnostic NLP model. Explicit negations
    such as 'no chest pain' are excluded to reduce obvious false matches.
    """
    text = (text or "").lower()
    found = []
    negation_words = ["no", "without", "denies", "denied", "not having"]

    for symptom_id, aliases in TEXT_SYMPTOM_ALIASES.items():
        for alias in aliases:
            pos = text.find(alias)
            if pos == -1:
                continue
            prefix = text[max(0, pos - 24):pos]
            if any(re.search(r"\b" + re.escape(word) + r"\s*$", prefix) for word in negation_words):
                continue
            found.append(symptom_id)
            break

    return sorted(set(found))


def evaluate(symptoms: List[str]) -> Dict[str, Any]:
    """Deterministic, auditable screening layer. It does not diagnose."""
    s = normalize(symptoms)
    red_flag_hits = []

    if "chest_pain" in s:
        _add_red_flag(red_flag_hits, "chest_pain")
    if "severe_shortness_of_breath" in s:
        _add_red_flag(red_flag_hits, "severe_shortness_of_breath")

    stroke_signs = {
        "one_sided_weakness", "speech_difficulty", "facial_droop",
        "sudden_vision_change", "sudden_loss_of_balance", "sudden_severe_headache"
    }
    if s.intersection(stroke_signs):
        _add_red_flag(red_flag_hits, "stroke_signs")

    if "confusion" in s or "fainting" in s:
        _add_red_flag(red_flag_hits, "confusion_or_fainting")

    dehydration_set = {
        "no_urination", "very_dark_urine", "rapid_heartbeat",
        "rapid_breathing", "confusion", "fainting"
    }
    if len(s.intersection(dehydration_set)) >= 2:
        _add_red_flag(red_flag_hits, "severe_dehydration_signs")

    hypotheses = []
    for condition in KB["conditions"]:
        supporting = set(condition["supporting_symptoms"])
        matched = sorted(s.intersection(supporting))
        if not matched:
            continue

        score = round(len(matched) / max(1, len(supporting)), 2)
        blocked = bool(s.intersection(set(condition.get("contraindicating_red_flags", []))))

        hypotheses.append({
            "id": condition["id"],
            "label": condition["label"],
            "score": score,
            "matched_symptoms": matched,
            "blocked_by_red_flag": blocked,
            "explanation": condition["explanation"],
            "source": condition["source"]
        })

    hypotheses.sort(key=lambda x: (x["blocked_by_red_flag"], -x["score"]))

    if any(rf["severity"] == "EMERGENCY" for rf in red_flag_hits):
        urgency = "EMERGENCY"
    elif any(rf["severity"] == "URGENT" for rf in red_flag_hits):
        urgency = "URGENT_SCREENING"
    elif hypotheses:
        urgency = "NON_EMERGENCY_SCREENING"
    else:
        urgency = "INSUFFICIENT_INFORMATION"

    return {
        "urgency": urgency,
        "red_flags": red_flag_hits,
        "hypotheses": hypotheses[:5],
        "input_symptoms": sorted(s),
        "kb_version": KB["version"]
    }


def extract_claims(text: str) -> List[Dict[str, Any]]:
    """Extract potentially verifiable claims using transparent pattern rules."""
    text = (text or "").strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    claims = []

    risky_terms = [
        "definitely", "certainly", "100%", "guaranteed", "confirmed",
        "you have ", "you are suffering", "this is ",
        "take ", "start ", "stop ", "antibiotic", "prescription",
        "dose", "dosage", "cure", "treat it at home", "manage it at home",
        "no need to see a doctor", "safe to ignore", "nothing serious", "just the flu"
    ]

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        low = sentence.lower()
        matched_terms = [term.strip() for term in risky_terms if term in low]
        claims.append({
            "text": sentence,
            "matched_terms": matched_terms,
            "potentially_risky": bool(matched_terms)
        })

    return claims


def _contains_any(text: str, phrases: List[str]) -> bool:
    return any(p in text for p in phrases)


def verify_draft(draft: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Independently verify the LLM draft against deterministic rules and KB."""
    draft = (draft or "").strip()
    lower = draft.lower()
    claims = extract_claims(draft)
    rejected = []
    accepted = []
    warnings = []

    # Safe negations prevent false positives such as "this is not a diagnosis".
    safe_disclaimers = [
        "not a diagnosis", "cannot diagnose", "cannot determine the diagnosis",
        "cannot confirm", "may be compatible with", "could be consistent with"
    ]

    certainty_phrases = [
        "definitely", "certainly", "100%", "guaranteed", "without a doubt",
        "confirmed diagnosis", "you absolutely have"
    ]

    treatment_phrases = [
        "start an antibiotic", "take an antibiotic", "start antibiotics",
        "take antibiotics", "take this medication", "start this medication",
        "stop your medication", "stop the medication", "prescription",
        "antibiotic dose", "medication dose", "dosage", "double your dose",
        "increase your dose", "decrease your dose", "cure it with"
    ]

    dismissal_phrases = [
        "treat it at home", "treat this at home", "manage it at home",
        "safe to stay home", "no need to see a doctor",
        "no medical attention is needed", "you don't need medical attention",
        "nothing serious", "safe to ignore", "just the flu", "just a cold"
    ]

    # 1) Certainty and treatment claims.
    for phrase in certainty_phrases:
        if phrase in lower:
            rejected.append({"claim": phrase, "reason": "Absolute medical certainty is not permitted."})

    for phrase in treatment_phrases:
        if phrase in lower:
            rejected.append({"claim": phrase, "reason": "Treatment, prescription, or dosing advice is outside the verified scope."})

    # 2) Definitive diagnosis claims. Do not reject generic "this is".
    diagnosis_phrases = [
        "you have influenza", "you have the flu", "you have covid", "you have pneumonia",
        "you have a urinary infection", "you have a uti", "you have a migraine",
        "you have food poisoning", "you have an infection",
        "you are suffering from influenza", "you are suffering from the flu",
        "you are suffering from pneumonia", "this is influenza", "this is the flu",
        "this is covid", "this is pneumonia", "this is a urinary infection",
        "this is a uti", "this is a migraine", "this is food poisoning", "this is an infection"
    ]
    for phrase in diagnosis_phrases:
        if phrase in lower:
            rejected.append({"claim": phrase, "reason": "Definitive diagnosis claim is not permitted by the prototype."})

    # 3) Compare named patterns against rule-engine evidence.
    allowed = result.get("hypotheses", [])
    allowed_text = " ".join(h["label"].lower() for h in allowed)
    known_labels = {h["label"].lower(): h for h in allowed}

    for h in KB["conditions"]:
        label = h["label"].lower()
        if label in lower and label not in known_labels:
            rejected.append({
                "claim": h["label"],
                "reason": "The LLM mentioned a knowledge-base pattern that was not supported by the current rule-engine result."
            })

    # 4) Emergency consistency.
    urgency = result.get("urgency", "INSUFFICIENT_INFORMATION")
    emergency_warning_phrases = [
        "urgent", "emergency", "immediate medical", "medical attention",
        "seek medical", "emergency department", "emergency care", "immediately"
    ]

    if urgency == "EMERGENCY":
        if not _contains_any(lower, emergency_warning_phrases):
            warnings.append("The rule engine detected an emergency red flag, but the draft does not clearly recommend urgent medical assessment.")
        for phrase in dismissal_phrases:
            if phrase in lower:
                rejected.append({"claim": phrase, "reason": "The draft conflicts with a deterministic emergency safety rule."})
                break

    # 5) Accept safe sentences that are not rejected.
    for item in claims:
        sentence = item["text"]
        if sentence.lower() in safe_disclaimers or any(p in sentence.lower() for p in safe_disclaimers):
            accepted.append(sentence)
        elif not item["potentially_risky"]:
            accepted.append(sentence)

    verified = len(rejected) == 0 and not (urgency == "EMERGENCY" and warnings)

    # Deterministic safety response always wins for emergency cases.
    if urgency == "EMERGENCY":
        final_text = build_safe_response(result)
        emergency_override = True
    elif verified:
        final_text = draft
        emergency_override = False
    else:
        final_text = build_safe_response(result)
        emergency_override = False

    return {
        "verified": verified,
        "accepted_claims": accepted,
        "rejected_claims": rejected,
        "warnings": warnings,
        "emergency_override": emergency_override,
        "final_text": final_text,
        "claim_count": len(claims)
    }


def build_safe_response(result: Dict[str, Any]) -> str:
    urgency = result.get("urgency")

    if urgency == "EMERGENCY":
        lines = [
            "EMERGENCY RED-FLAG PATTERN DETECTED.",
            "The deterministic safety rules identified a red-flag symptom pattern.",
            "This prototype cannot determine the cause. Urgent/emergency medical assessment is recommended."
        ]
        for rf in result.get("red_flags", []):
            lines.append(f"• {rf['message']}")
        return "\n\n".join(lines)

    if urgency == "URGENT_SCREENING":
        lines = [
            "URGENT SAFETY PATTERN DETECTED.",
            "The rule engine identified a possible urgent concern. This prototype cannot diagnose the cause, and prompt medical assessment may be needed."
        ]
        for rf in result.get("red_flags", []):
            lines.append(f"• {rf['message']}")
        return "\n\n".join(lines)

    hypotheses = result.get("hypotheses", [])
    if hypotheses:
        lines = ["This is a screening result, not a diagnosis.", "Possible patterns supported by the rule engine:"]
        for h in hypotheses[:3]:
            lines.append(f"• {h['label']} — score {h['score']:.2f}; matched: {', '.join(h['matched_symptoms'])}")
        lines.append("The pattern should be interpreted with the person's full clinical context. Professional evaluation is appropriate if symptoms are severe, persistent, or worsening.")
        return "\n".join(lines)

    return (
        "The available symptoms do not match a sufficiently specific rule in the current knowledge base. "
        "This prototype should not guess a diagnosis. More information or professional medical assessment is needed."
    )
