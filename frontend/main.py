# main.py (Streamlit UI)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import streamlit as st
from pathlib import Path
from app.chunking import preprocess_book
from app.embedder import build_vector_store, load_embedding_model
from app.ontology import run_ontology_extraction, load_llm_pipeline as load_ontology_llm
from app.retriever import retrieve_top_k_chunks
from app.generator import generate_answer, load_llm_pipeline as load_rag_llm

st.set_page_config(page_title="Ontology + RAG QA", layout="wide")
st.title("📚 Ontology-Based Semantic Search (RAG)")
st.markdown("Upload a `.txt` file, extract ontology, and query using RAG.")

# Sidebar toggle for model selection
st.sidebar.markdown("## ⚙️ Model Settings")
dev_mode = st.sidebar.toggle("Use Dev Mode (flan-t5-base)", value=True)
st.sidebar.markdown(f"**Current Model:** {'flan-t5-base 🧪 (dev)' if dev_mode else 'flan-t5-large 🚀 (prod)'}")

# Upload text file
uploaded_file = st.file_uploader("Upload a plain text file", type=["txt"])
if uploaded_file:
    file_path = Path("data/raw") / uploaded_file.name
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ Book uploaded successfully.")

    chunks = preprocess_book(file_path)
    st.write(f"Book has been chunked into {len(chunks)} segments.")

    # Build vector store
    with st.spinner("📦 Building vector store..."):
        embed_model = load_embedding_model()
        build_vector_store(chunks, embedding_model=embed_model, db_name="book_index")
        st.success("✅ Vector store created.")

    # Ontology extraction
    with st.spinner("🧠 Extracting ontology from content..."):
        ontology_llm = load_ontology_llm(dev_mode=dev_mode)
        sample_chunks = chunks[:1]  # limit to 1 for speed
        combined_ontology = {"classes": [], "entities": {}, "relationships": []}

        for i, chunk in enumerate(sample_chunks, 1):
            st.write(f"🧠 Extracting ontology from chunk {i}...")
            ontology = run_ontology_extraction(chunk, pipe=ontology_llm)
            st.write(f"✅ Finished chunk {i}")
            if ontology:
                combined_ontology["classes"].extend(ontology.get("classes", []))
                for cls, ents in ontology.get("entities", {}).items():
                    combined_ontology["entities"].setdefault(cls, []).extend(ents)
                combined_ontology["relationships"].extend(ontology.get("relationships", []))

        st.subheader("📘 Extracted Ontology (Sample)")
        st.json(combined_ontology)

# Question Answering Interface
st.subheader("💬 Ask a question about the book")
user_query = st.text_input("Enter your question here:")

if user_query:
    with st.spinner("🔍 Retrieving answer..."):
        top_chunks = retrieve_top_k_chunks(user_query, k=8)
        rag_llm = load_rag_llm(dev_mode=dev_mode)
        answer = generate_answer(user_query, top_chunks, pipe=rag_llm)

        st.markdown("### ✅ Answer")
        st.success(answer)

        with st.expander("🔎 Retrieved Context Chunks"):
            for i, chunk in enumerate(top_chunks, 1):
                st.markdown(f"**Chunk {i}:**")
                st.code(chunk[:500])
