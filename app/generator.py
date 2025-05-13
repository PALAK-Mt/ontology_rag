from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    pipeline,
)
from typing import List

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

def format_prompt(query: str, context_chunks: List[str]) -> str:
    # Truncate context to avoid exceeding model input limit
    context = "\n\n".join(context_chunks)
    context = context[:2000]  # ~500 tokens worth of context

    prompt = (
        "You are a helpful assistant. Use the context below to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    return prompt

def generate_answer(query: str, context_chunks: List[str], pipe=None) -> str:
    if pipe is None:
        pipe = load_llm_pipeline()

    prompt = format_prompt(query, context_chunks)
    response = pipe(prompt, max_new_tokens=600, do_sample=False)
    return response[0]["generated_text"].split("Answer:")[-1].strip()