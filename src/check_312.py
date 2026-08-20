import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

chunks = json.loads(Path("data/processed/chunks.json").read_text(encoding="utf-8"))

print("=== Chunks for § 164.312 ===")
hits = [c for c in chunks if c["section"] == "§ 164.312"]
print(f"count: {len(hits)}")
for c in hits:
    print(f"\n  {c['id']} ({c['chars']} chars)")
    print(f"  {c['text'][:300]}")

print("\n=== Which chunks mention 'encrypt' at all? ===")
enc = [c for c in chunks if "encrypt" in c["text"].lower()]
print(f"count: {len(enc)}")
for c in enc[:10]:
    print(f"  {c['id']:<14} {c['section']:<12} {c['title'][:45]}")

print("\n=== Where does § 164.312 rank for the query? ===")
model = SentenceTransformer("all-MiniLM-L6-v2")
col = chromadb.PersistentClient(path="chroma_db").get_collection("hipaa")
qv = model.encode(["Do I need to encrypt patient data at rest?"])[0].tolist()
res = col.query(query_embeddings=[qv], n_results=40)

for rank, (cid, meta, dist) in enumerate(
    zip(res["ids"][0], res["metadatas"][0], res["distances"][0]), 1
):
    if meta["section"] in ("§ 164.312", "§ 164.306") or "encrypt" in res["documents"][0][rank-1].lower():
        print(f"  rank {rank:>3}  [{dist:.3f}]  {cid:<14} {meta['section']}")