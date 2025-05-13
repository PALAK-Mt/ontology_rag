import numpy as np
from app.embedder import load_embedding_model, load_vector_store


def embed_query(query: str, embedding_model) -> np.ndarray:
    """
    Converts a user query into an embedding vector.
    """
    return embedding_model.encode([query])


def retrieve_top_k_chunks(query: str, k: int = 5, db_name: str = "book_index"):
    """
    Given a user query, retrieve top-k most relevant chunks from the vector DB.
    """
    index, chunks = load_vector_store(db_name)
    embedding_model = load_embedding_model()

    query_vec = embed_query(query, embedding_model)
    distances, indices = index.search(np.array(query_vec), k)

    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks
