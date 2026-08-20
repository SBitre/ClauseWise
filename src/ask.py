import os
import sys

import chromadb
from dotenv import load_dotenv
from google import genai
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from search import build_bm25, search, MODEL_NAME, DB_DIR, COLLECTION

load_dotenv()

MAX_DISTANCE = 0.75          # beyond this, treat the corpus as not covering it
MODEL = "gemini-3.7-flash"

PROMPT = """You are a HIPAA compliance assistant. Answer ONLY from the excerpts below.

RULES:
- Use no outside knowledge. If the excerpts do not answer the question, reply
  exactly: "I don't know — the provided HIPAA documents don't cover this."
- Cite the section number after each claim, like (§ 164.404).
- Quote the regulation's own wording for requirements where possible.
- Be concise.

EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""


def main():
    question = " ".join(sys.argv[1:]) or "How long do I have to notify individuals after a breach?"

    model = SentenceTransformer(MODEL_NAME)
    collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)
    bm25, bm25_ids = build_bm25(collection)

    hits = search(collection, model, bm25, bm25_ids, question)

    # Grounding gate: if nothing is semantically close, don't call the LLM at all.
    closest = min((h["distance"] for h in hits if h["distance"] is not None), default=1.0)
    if closest > MAX_DISTANCE:
        print(f"Q: {question}\n")
        print("I don't know — the provided HIPAA documents don't cover this.")
        print(f"\n(closest match {closest:.3f} exceeded threshold {MAX_DISTANCE})")
        return

    context = "\n\n".join(
        f"[{h['section']} {h['title']}]\n{h['text']}" for h in hits
    )

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    interaction = client.interactions.create(
        model=MODEL,
        input=PROMPT.format(context=context, question=question),
    )

    print(f"Q: {question}\n")
    print(interaction.output_text)
    print("\n--- retrieved ---")
    for h in hits:
        d = f"{h['distance']:.3f}" if h["distance"] is not None else "bm25"
        print(f"  [{d}] {h['section']} {h['title'][:50]}")


if __name__ == "__main__":
    main()