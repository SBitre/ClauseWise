import re
import sys
from collections import defaultdict

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DB_DIR = "chroma_db"
COLLECTION = "hipaa"
MODEL_NAME = "all-MiniLM-L6-v2"
CANDIDATES = 30
TOP_K = 8
RRF_K = 60

DEFAULT_QUERIES = [
    "Do I need to encrypt patient data at rest?",
    "How long do I have to notify individuals after a breach?",
    "Who counts as a business associate?",
    "What is the best pizza topping?",
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]{3,}", text.lower())


def build_bm25(collection):
    """BM25 down-weights common terms, so 'data' counts far less than 'encrypt'."""
    all_docs = collection.get(include=["documents", "metadatas"])
    ids = all_docs["ids"]
    corpus = [
        tokenize(f"{m['section']} {m['title']} {d}")
        for d, m in zip(all_docs["documents"], all_docs["metadatas"])
    ]
    return BM25Okapi(corpus), ids


def search(collection, model, bm25, bm25_ids, query: str):
    qv = model.encode([query])[0].tolist()
    dense = collection.query(query_embeddings=[qv], n_results=CANDIDATES)
    dense_ids = dense["ids"][0]
    dist_by_id = dict(zip(dense_ids, dense["distances"][0]))

    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(zip(scores, bm25_ids), reverse=True)
    sparse_ids = [cid for s, cid in ranked[:CANDIDATES] if s > 0]

    fused = defaultdict(float)
    for rank, cid in enumerate(dense_ids, 1):
        fused[cid] += 1 / (RRF_K + rank)
    for rank, cid in enumerate(sparse_ids, 1):
        fused[cid] += 1 / (RRF_K + rank)

    top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K]
    got = collection.get(ids=[c for c, _ in top], include=["documents", "metadatas"])
    by_id = {i: (d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])}

    return [
        {
            "id": cid,
            "section": by_id[cid][1]["section"],
            "title": by_id[cid][1]["title"],
            "text": by_id[cid][0],
            "distance": dist_by_id.get(cid),
            "fused": f,
        }
        for cid, f in top
    ]


def main():
    model = SentenceTransformer(MODEL_NAME)
    collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)
    bm25, bm25_ids = build_bm25(collection)

    queries = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEFAULT_QUERIES

    for q in queries:
        print("=" * 72)
        print(f"Q: {q}\n")
        for r in search(collection, model, bm25, bm25_ids, q):
            d = f"{r['distance']:.3f}" if r["distance"] is not None else "  -  "
            print(f"  [d={d} f={r['fused']:.4f}]  {r['section']}  {r['title'][:45]}")
            print(f"      {r['text'][:170].strip()}...\n")


if __name__ == "__main__":
    main()