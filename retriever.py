import numpy as np
import faiss
import pickle
import streamlit as st
from sentence_transformers import SentenceTransformer

INDEX_PATH  = "faiss_index.bin"
CHUNKS_PATH = "chunks.pkl"
MODEL_NAME  = "BAAI/bge-m3"

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource
def load_index():
    return faiss.read_index(INDEX_PATH)

@st.cache_resource
def load_chunks():
    with open(CHUNKS_PATH, "rb") as f:
        return pickle.load(f)

def retrieve(query: str, top_k: int = 5, threshold: float = 0.40):
    model  = load_model()
    index  = load_index()
    chunks = load_chunks()

    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype="float32")

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if score < threshold:
            continue
        chunk = chunks[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)

    return results