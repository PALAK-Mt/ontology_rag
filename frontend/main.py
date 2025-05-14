import sys
import re
from pathlib import Path
import streamlit as st

# Allow importing from app/
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.chunking import preprocess_book
from app.embedder import build_vector_store, load_embedding_model
from app.ontology import run_ontology_extraction
from app.retriever import retrieve_top_k_chunks
from app.generator import generate_answer


def extract_metadata(raw_text: str) -> dict:
    title_match = re.search(r"Title:\s*(.*)", raw_text)
    author_match = re.search(r"Author:\s*(.*)", raw_text)
    return {
        "title": title_match.group(1).strip() if title_match else "Unknown",
        "author": author_match.group(1).strip() if author_match else "Unknown"
    }

# --- Session state init ---
if "book_uploaded" not in st.session_state:
    st.session_state.book_uploaded = False
    st.session_state.chunks = []
    st.session_state.ontology = {}
    st.session_state.people = set()
    st.session_state.metadata = {}

# --- UI Setup ---
st.set_page_config(page_title="Ontology + RAG QA", layout="wide")
st.title("📚 Ontology-Based Semantic Search (RAG)")
st.markdown("Upload a `.txt` file, extract ontology, and query using RAG.")

st.sidebar.markdown("## ⚙️ Model Settings")
st.sidebar.markdown("Model: Hugging Face Hosted LLM")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a plain text file", type=["txt"])

if uploaded_file and not st.session_state.book_uploaded:
    file_path = Path("data/raw") / uploaded_file.name
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ Book uploaded successfully.")

    raw_text = file_path.read_text(encoding="utf-8")
    st.session_state.metadata = extract_metadata(raw_text)

    chunks = preprocess_book(file_path)
    st.session_state.chunks = chunks
    st.success(f"📚 Book has been chunked into {len(chunks)} segments.")

    # Vector DB
    with st.spinner("📦 Building vector store..."):
        embed_model = load_embedding_model()
        build_vector_store(chunks, embedding_model=embed_model, db_name="book_index")
        st.success("✅ Vector store created.")

    # Ontology extraction (just a few chunks)
    with st.spinner("🧠 Extracting ontology from sample chunks..."):
        sample_chunks = chunks[:8]
        combined = {"classes": [], "entities": {}, "relationships": []}

        for i, chunk in enumerate(sample_chunks, 1):
            st.write(f"🧠 Extracting ontology from chunk {i}...")
            ontology = run_ontology_extraction(chunk)
            st.write(f"✅ Finished chunk {i}")
            if ontology:
                combined["classes"].extend(ontology.get("classes", []))
                for cls, ents in ontology.get("entities", {}).items():
                    normalized_cls = cls.strip().title()
                    combined["entities"].setdefault(normalized_cls, []).extend(ents)
                combined["relationships"].extend(ontology.get("relationships", []))

        for cls in combined["entities"]:
            combined["entities"][cls] = list(set(combined["entities"][cls]))

        st.session_state.ontology = combined
        st.session_state.people = {
            ent for cls, ents in combined["entities"].items()
            if cls.lower() == "person" or "character" in cls.lower()
            for ent in ents
        }

        st.subheader("📘 Extracted Ontology (Sample)")
        st.json(combined)

    st.session_state.book_uploaded = True

# --- QA Interface ---
if st.session_state.book_uploaded:
    st.subheader("💬 Ask a question about the book")
    user_query = st.text_input("Enter your question:")

    if user_query:
        q_lower = user_query.lower()
        meta = st.session_state.metadata

        if "author" in q_lower or "title" in q_lower:
            st.markdown("### ℹ️ Book Metadata")
            if "author" in q_lower:
                st.success(f"**Author:** {meta.get('author', 'Unknown')}")
            elif "title" in q_lower:
                st.success(f"**Title:** {meta.get('title', 'Unknown')}")
        elif "character" in q_lower or "people" in q_lower or "person" in q_lower:
            if st.session_state.people:
                st.markdown("### 👥 Characters Found")
                st.success(", ".join(sorted(st.session_state.people)))
            else:
                st.warning("No characters found. Try increasing chunk coverage.")
        else:
                
            with st.spinner("🔍 Answering using RAG..."):
                top_chunks = retrieve_top_k_chunks(user_query, k=8)

                # 🔐 Contextual relevance check
                embed_model = load_embedding_model()
                from sentence_transformers.util import cos_sim

                query_vec = embed_model.encode([user_query])
                chunk_vecs = embed_model.encode(top_chunks)
                sims = cos_sim(query_vec, chunk_vecs)[0]
                max_sim = max(sims).item()

                if max_sim < 0.3:  # 🔍 Threshold: adjust for stricter filtering
                    st.warning("🤔 I couldn't find relevant information in the book to answer that.")
                else:
                    all_triples = st.session_state.ontology.get("relationships", [])
                    q_lower = user_query.lower()
                    ontology_triples = [
                        t for t in all_triples if isinstance(t, (list, tuple)) and len(t) == 3 and
                        any(term.lower() in q_lower or q_lower in term.lower() for term in t)
                    ]

                    answer = generate_answer(user_query, top_chunks, ontology_triples=ontology_triples)
                    st.markdown("### ✅ Answer")
                    st.success(answer)

                    with st.expander("🔎 Retrieved Context"):
                        for i, chunk in enumerate(top_chunks, 1):
                            st.markdown(f"**Chunk {i}**")
                            st.code(chunk[:500])

