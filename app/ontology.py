# app/ontology.py
from pathlib import Path
import json
from app.llm_chat import chat_with_model

PROMPT_PATH = Path("prompts/ontology_prompt.txt")

def load_prompt(chunk_text: str) -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read().format(text=chunk_text)

def run_ontology_extraction(chunk_text: str) -> dict:
    prompt = load_prompt(chunk_text)

    response_text = chat_with_model([
        {"role": "user", "content": prompt}
    ])

    json_start = response_text.find("{")
    json_output = response_text[json_start:]

    try:
        return json.loads(json_output)
    except json.JSONDecodeError:
        print("⚠️ Could not parse ontology JSON.")
        print(response_text)
        return {}
