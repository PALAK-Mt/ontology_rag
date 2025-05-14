# app/generator.py
from typing import List, Optional
from app.llm_chat import chat_with_model

def generate_answer(query: str, context_chunks: List[str], ontology_triples: Optional[List[List[str]]] = None) -> str:
    context = "\n\n".join(context_chunks)[:3000]

    triple_section = ""
    if ontology_triples:
        
        triple_lines = [
    f"{a} — {b} — {c}" 
    for t in ontology_triples 
    if isinstance(t, (list, tuple)) and len(t) == 3
    for a, b, c in [t]
]

        triple_section = "Ontology Knowledge Triples:\n" + "\n".join(triple_lines) + "\n\n"

    system = "You are a helpful assistant that answers questions using the provided context and ontology facts."
    user = f"{triple_section}Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    response_text = chat_with_model([
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ])
    return response_text.strip()
