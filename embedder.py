import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

KNOWLEDGE_BASE_PATH = "knowledge_base.json"
INDEX_PATH          = "faiss_index.bin"
CHUNKS_PATH         = "chunks.pkl"
MODEL_NAME          = "BAAI/bge-m3"

def build_chunks(data):
    chunks = []
    for disease in data["diseases"]:
        name_en = disease["name_en"]
        name_ar = disease["name_ar"]

        # One chunk per section, both languages together
        sections = [
            {
                "section": "description",
                "text_en": disease.get("description_en", ""),
                "text_ar": disease.get("description_ar", ""),
            },
            {
                "section": "symptoms",
                "text_en": " ".join(disease.get("symptoms_en", [])),
                "text_ar": " ".join(disease.get("symptoms_ar", [])),
            },
            {
                "section": "causes",
                "text_en": " ".join(disease.get("causes_en", [])),
                "text_ar": " ".join(disease.get("causes_ar", [])),
            },
            {
                "section": "severity",
                "text_en": disease.get("severity_en", ""),
                "text_ar": disease.get("severity_ar", ""),
            },
            {
                "section": "when_to_see_doctor",
                "text_en": disease.get("when_to_see_doctor_en", ""),
                "text_ar": disease.get("when_to_see_doctor_ar", ""),
            },
        ]

        for s in sections:
            # Skip empty chunks
            if not s["text_en"] and not s["text_ar"]:
                continue

            # Text used for embedding — both languages so bge-m3 can match either
            embed_text = f"{name_en} {s['section']}: {s['text_en']} {s['text_ar']}"

            chunks.append({
                "disease_id":   disease["id"],
                "disease_en":   name_en,
                "disease_ar":   name_ar,
                "section":      s["section"],
                "text_en":      s["text_en"],
                "text_ar":      s["text_ar"],
                "severity":     disease.get("severity_level", "none"),
                "embed_text":   embed_text,
            })

    return chunks


def build_index():
    print("Loading knowledge base...")
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Building chunks...")
    chunks = build_chunks(data)
    print(f"  → {len(chunks)} chunks created")

    print("Loading embedding model (first run downloads ~570MB)...")
    model = SentenceTransformer(MODEL_NAME)

    print("Embedding chunks...")
    texts = [c["embed_text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity (embeddings are normalized)
    index.add(embeddings)

    print("Saving index and chunks...")
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Done! {len(chunks)} chunks indexed.")


if __name__ == "__main__":
    build_index()