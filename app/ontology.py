from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    pipeline,
)
from pathlib import Path
import json

PROMPT_PATH = Path("prompts/ontology_prompt.txt")

def load_prompt(chunk_text: str) -> str:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    return prompt_template.format(text=chunk_text)

def load_llm_pipeline(dev_mode=True):
    if dev_mode:
        model_id = "google/flan-t5-base"
        task = "text2text-generation"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    else:
        model_id = "openchat/openchat-3.5-1210"
        task = "text-generation"
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return pipeline(task, model=model, tokenizer=tokenizer)

def run_ontology_extraction(chunk_text: str, pipe=None) -> dict:
    if pipe is None:
        pipe = load_llm_pipeline()

    # Limit the input text to prevent token overflow
    chunk_text = chunk_text[:1200]  # ~500 tokens depending on content

    prompt = load_prompt(chunk_text)
    response = pipe(prompt, max_new_tokens=800, do_sample=False)
    raw_output = response[0]["generated_text"]

    json_start = raw_output.find("{")
    json_output = raw_output[json_start:]

    try:
        ontology = json.loads(json_output)
    except json.JSONDecodeError:
        print("⚠️ Could not parse LLM output as JSON. Raw output:")
        print(raw_output)
        ontology = {}

    return ontology
