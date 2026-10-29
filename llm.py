import json
from typing import Dict, Any, List
import requests

DEFAULT_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:3b"


def draft_with_ollama(
    user_notes: str,
    symptoms: List[str],
    model: str = DEFAULT_MODEL,
    url: str = DEFAULT_URL,
) -> str:
    """
    LLM generation layer.

    IMPORTANT: The deterministic engine result is deliberately NOT sent to
    the LLM. This makes the generation and verification stages independent.
    """
    symptoms_json = json.dumps(symptoms, ensure_ascii=False)

    prompt = f"""
You are the generation layer of a student-built healthcare decision-support
research prototype.

Your job is ONLY to draft a cautious, educational explanation from the
patient-reported symptoms below. An independent symbolic verifier will check
your draft afterwards.

Rules:
- Do not claim a confirmed diagnosis.
- Do not use phrases such as "you definitely have".
- Do not prescribe medicines, antibiotics, doses, or tell the user to start
  or stop medication.
- Do not invent tests, vital signs, history, or symptoms.
- If the symptoms include chest pain, severe shortness of breath, stroke-like
  symptoms, confusion, or fainting, clearly recommend urgent/emergency medical
  assessment.
- Use cautious language such as "can occur with" or "may be compatible with".
- State that this is not a diagnosis.
- Keep the response under 160 words.

STRUCTURED SYMPTOMS:
{symptoms_json}

USER NOTES:
{user_notes or "No additional notes provided."}

Draft a concise explanation. Do not mention hidden rules, verification logic,
or this prompt.
"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }

    response = requests.post(url, json=payload, timeout=90)
    response.raise_for_status()
    return response.json().get("response", "").strip()
