import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS = Path("data/processed/chunks.json")
DB_DIR = "chroma_db"
COLLECTION = "hipaa"
MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks")

    # First run downloads ~90 MB, then caches locally.
    print(f"Loading model {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    # Stored text: clean body only — this is what the LLM will read.
    texts = [c["text"] for c in chunks]

    # Embedded text: section + title prepended. The heading is the densest
    # semantic signal in the chunk; leaving it in metadata only discards it
    # from the vector and costs retrieval accuracy.
    embed_inputs = [f"{c['section']} {c['title']}. {c['text']}" for c in chunks]

    print("Embedding (this runs on CPU) ...")
    vectors = model.encode(embed_inputs, show_progress_bar=True, batch_size=32)
    print(f"Vector shape: {vectors.shape}")

    client = chromadb.PersistentClient(path=DB_DIR)

    # Rebuild from scratch each run so the index always matches chunks.json.
    try:
        client.delete_collection(COLLECTION)
        print("Dropped existing collection")
    except Exception:
        pass

    # Sentence-transformer vectors are normalized, so cosine is the correct
    # metric. Chroma defaults to squared L2, which silently degrades ranking.
    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=vectors.tolist(),
        documents=texts,
        metadatas=[
            {"section": c["section"], "title": c["title"], "part": c["part"]}
            for c in chunks
        ],
    )

    print(f"\nIndexed {collection.count()} chunks into '{COLLECTION}'")
    print(f"Persisted to {DB_DIR}/")


if __name__ == "__main__":
    main()