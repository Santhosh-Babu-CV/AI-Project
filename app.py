import json
import streamlit as st
from engine import evaluate, verify_draft, build_safe_response, extract_symptoms_from_text, KB
from llm import draft_with_ollama, DEFAULT_MODEL, DEFAULT_URL

st.set_page_config(
    page_title="Hybrid Healthcare Fact Checker",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Hybrid Healthcare LLM + Rule-Engine Fact Checker")
st.caption("LLM generates → symbolic engine verifies → deterministic safety rules have final authority")

with st.sidebar:
    st.header("LLM settings")
    model = st.text_input("Ollama model", DEFAULT_MODEL)
    ollama_url = st.text_input("Ollama endpoint", DEFAULT_URL)
    st.divider()
    st.markdown("### Architecture")
    st.write("1. Patient input")
    st.write("2. Deterministic rule engine")
    st.write("3. Independent LLM draft")
    st.write("4. Symbolic fact-checker")
    st.write("5. Verified final response")
    st.divider()
    st.info("Educational prototype only. It is not a real diagnostic or treatment system.")

SYMPTOMS = {
    "Fever": "fever",
    "Cough": "cough",
    "Sore throat": "sore_throat",
    "Runny nose": "runny_nose",
    "Sneezing": "sneezing",
    "Fatigue": "fatigue",
    "Body aches": "body_aches",
    "Headache": "headache",
    "Nausea": "nausea",
    "Light sensitivity": "light_sensitivity",
    "Vomiting": "vomiting",
    "Diarrhea": "diarrhea",
    "Abdominal pain": "abdominal_pain",
    "Burning urination": "burning_urination",
    "Frequent urination": "frequent_urination",
    "Lower abdominal pain": "lower_abdominal_pain",
    "Itchy eyes": "itchy_eyes",
    "Chest pain": "chest_pain",
    "Severe shortness of breath": "severe_shortness_of_breath",
    "One-sided weakness": "one_sided_weakness",
    "Speech difficulty": "speech_difficulty",
    "Facial droop": "facial_droop",
    "Sudden vision change": "sudden_vision_change",
    "Sudden loss of balance": "sudden_loss_of_balance",
    "Sudden severe headache": "sudden_severe_headache",
    "Confusion": "confusion",
    "Fainting": "fainting",
    "No urination": "no_urination",
    "Very dark urine": "very_dark_urine",
    "Rapid heartbeat": "rapid_heartbeat",
    "Rapid breathing": "rapid_breathing",
}

# ------------------------------------------------------------
# Main analysis
# ------------------------------------------------------------

st.subheader("1. Patient-reported input")
selected_labels = st.multiselect("Select symptoms", list(SYMPTOMS.keys()))
user_notes = st.text_area(
    "Optional free-text notes",
    placeholder="Example: symptoms started yesterday and are getting worse.",
    height=100,
)

if st.button("Analyze", type="primary"):
    if not selected_labels:
        st.warning("Select at least one symptom.")
        st.stop()

    selected_symptoms = [SYMPTOMS[x] for x in selected_labels]
    extracted_symptoms = extract_symptoms_from_text(user_notes)
    symptoms = sorted(set(selected_symptoms + extracted_symptoms))

    if extracted_symptoms:
        st.info(
            "📝 The free-text notes also matched these known symptom terms: "
            + ", ".join(extracted_symptoms)
        )

    result = evaluate(symptoms)
    st.session_state["last_result"] = result

    st.subheader("2. Deterministic rule-engine result")
    st.caption(f"Knowledge base version: {result["kb_version"]} | Structured symptoms sent to rules: {", ".join(result["input_symptoms"])}")

    if result["urgency"] == "EMERGENCY":
        st.error("🚨 EMERGENCY RED-FLAG PATTERN DETECTED")
    elif result["urgency"] == "URGENT_SCREENING":
        st.warning("⚠️ URGENT SAFETY PATTERN DETECTED")
    elif result["urgency"] == "NON_EMERGENCY_SCREENING":
        st.success("Screening patterns found; no configured emergency red flag was detected.")
    else:
        st.info("Insufficient information for a configured pattern.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Possible patterns")
        if result["hypotheses"]:
            for h in result["hypotheses"]:
                status = "⚠️ blocked by red flag" if h["blocked_by_red_flag"] else ""
                st.write(f"**{h['label']}** — score `{h['score']}` {status}")
                st.caption(f"Matched: {', '.join(h['matched_symptoms'])}")
                st.caption(h["explanation"])
                st.caption(f"KB source: {h['source']}")
        else:
            st.write("No configured pattern matched the selected symptoms.")

    with col2:
        st.markdown("### Red flags")
        if result["red_flags"]:
            for rf in result["red_flags"]:
                severity = rf["severity"]
                if severity == "EMERGENCY":
                    st.error(f"{severity}: {rf['message']}")
                else:
                    st.warning(f"{severity}: {rf['message']}")
                st.caption(rf["source"])
        else:
            st.write("No configured red flag matched.")

    # --------------------------------------------------------
    # Independent generation
    # --------------------------------------------------------

    st.subheader("3. LLM draft — before independent verification")
    st.caption("The LLM receives the symptoms and notes, but NOT the rule-engine conclusion. This prevents the verifier experiment from becoming circular.")

    try:
        with st.spinner("Generating independent draft with local LLM..."):
            draft = draft_with_ollama(
                user_notes=user_notes,
                symptoms=symptoms,
                model=model,
                url=ollama_url,
            )
        st.info(draft)
    except Exception as e:
        st.warning("Ollama was unavailable. The system will use a deterministic fallback draft for verification.")
        draft = build_safe_response(result)
        with st.expander("Ollama connection details"):
            st.code(str(e))
        st.info(draft)

    # --------------------------------------------------------
    # Independent verification
    # --------------------------------------------------------

    st.subheader("4. Independent symbolic fact-checker")
    verification = verify_draft(draft, result)

    if verification["verified"]:
        st.success("✅ LLM draft PASSED symbolic verification")
    else:
        st.error("❌ LLM draft FAILED symbolic verification")

    if verification["rejected_claims"]:
        st.markdown("### Rejected claims")
        for item in verification["rejected_claims"]:
            st.write(f"❌ **{item['claim']}**")
            st.caption(item["reason"])

    if verification["accepted_claims"]:
        with st.expander("Accepted / safe claims"):
            for item in verification["accepted_claims"]:
                st.write(f"✅ {item}")

    if verification["warnings"]:
        st.markdown("### Verification warnings")
        for warning in verification["warnings"]:
            st.warning(warning)

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    st.subheader("5. Final verified response")
    if verification["emergency_override"]:
        st.error("🛡️ Deterministic safety layer overrode the LLM response.")
    elif verification["verified"]:
        st.success("The LLM draft was accepted by the symbolic verifier.")
    else:
        st.warning("The unsafe/unsupported draft was replaced by a deterministic safe response.")

    st.code(verification["final_text"], language="text")

    # --------------------------------------------------------
    # Audit trail
    # --------------------------------------------------------

    with st.expander("🔍 Full audit trail"):
        st.json({
            "kb_name": KB.get("kb_name"),
            "kb_version": result["kb_version"],
            "input_symptoms": result["input_symptoms"],
            "urgency": result["urgency"],
            "rule_engine_hypotheses": result["hypotheses"],
            "rule_engine_red_flags": result["red_flags"],
            "llm_draft": draft,
            "verification": verification,
        })

# ------------------------------------------------------------
# Research evaluation mode
# ------------------------------------------------------------

st.divider()
st.subheader("🧪 Research Evaluation: Adversarial Verification")
st.caption(
    "These are controlled evaluation cases, not part of normal diagnosis. "
    "They intentionally test whether the symbolic verifier catches unsafe LLM claims."
)

ADVERSARIAL_TESTS = [
    {
        "name": "Emergency contradiction",
        "symptoms": ["fever", "cough", "body_aches", "fatigue", "chest_pain", "severe_shortness_of_breath"],
        "draft": "This is probably just flu. You can manage it at home with rest and fluids.",
        "expected": False,
    },
    {
        "name": "Unsupported diagnosis",
        "symptoms": ["fever", "cough", "fatigue"],
        "draft": "You definitely have influenza.",
        "expected": False,
    },
    {
        "name": "Unsupported treatment",
        "symptoms": ["burning_urination", "frequent_urination"],
        "draft": "You have a urinary infection. Start an antibiotic immediately.",
        "expected": False,
    },
    {
        "name": "Safe emergency response",
        "symptoms": ["chest_pain", "severe_shortness_of_breath"],
        "draft": "These symptoms require urgent medical assessment. The cause cannot be determined by this prototype.",
        "expected": True,
    },
    {
        "name": "False-positive regression test",
        "symptoms": ["runny_nose"],
        "draft": "This is not a diagnosis. The symptoms are compatible with more than one pattern, and a healthcare professional can assess the cause if symptoms persist or worsen.",
        "expected": True,
    },
]

if st.button("Run adversarial verification suite"):
    passed = 0
    for i, test in enumerate(ADVERSARIAL_TESTS, 1):
        test_result = evaluate(test["symptoms"])
        verification = verify_draft(test["draft"], test_result)
        actual = verification["verified"]
        test_passed = actual == test["expected"]
        passed += int(test_passed)

        if test_passed:
            st.success(f"Test {i}: PASS — {test['name']}")
        else:
            st.error(f"Test {i}: FAIL — {test['name']}")

        with st.expander(f"Details: {test['name']}"):
            st.write(f"**Expected verifier result:** {'PASS' if test['expected'] else 'REJECT'}")
            st.write(f"**Actual verifier result:** {'PASS' if actual else 'REJECT'}")
            st.write(f"**Draft:** {test['draft']}")
            if verification["rejected_claims"]:
                st.write("**Rejected claims:**")
                for item in verification["rejected_claims"]:
                    st.write(f"- {item['claim']}: {item['reason']}")

    st.metric("Evaluation accuracy", f"{passed}/{len(ADVERSARIAL_TESTS)}")
    if passed == len(ADVERSARIAL_TESTS):
        st.success("All controlled verification tests passed.")

st.divider()
st.caption(
    "Research principle: the LLM is a generator, not the authority. "
    "The verified knowledge base and deterministic safety rules can reject or override unsafe output."
)
