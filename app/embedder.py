import os
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pickle

VECTOR_DB_DIR = Path("data/vector_store")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    """
    Load a sentence transformer embedding model.
    """
    return SentenceTransformer(model_name)


def build_vector_store(chunks, embedding_model=None, db_name="book_index"):
    """
    Generate embeddings for chunks and build a FAISS vector store.
    Also saves index and metadata to disk.
    """
    if embedding_model is None:
        embedding_model = load_embedding_model()

    print("[🔁] Generating embeddings...")
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    # Save FAISS index
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(VECTOR_DB_DIR / f"{db_name}.faiss"))

    # Save metadata (chunks)
    with open(VECTOR_DB_DIR / f"{db_name}_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"[✅] Stored {len(chunks)} chunks in FAISS DB.")
    return index


def load_vector_store(db_name="book_index"):
    """
    Load FAISS index and associated metadata.
    """
    index_path = VECTOR_DB_DIR / f"{db_name}.faiss"
    chunks_path = VECTOR_DB_DIR / f"{db_name}_chunks.pkl"

    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError("Vector DB files not found.")

    index = faiss.read_index(str(index_path))

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks
