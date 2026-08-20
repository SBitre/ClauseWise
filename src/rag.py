"""Core RAG engine. Loaded once, serves many requests."""

import os
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field

import chromadb
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()

# Anchor to the project root so the app works from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = str(PROJECT_ROOT / "chroma_db")
COLLECTION = "hipaa"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-3.7-flash"

CANDIDATES = 30          # depth each retriever searches
TOP_K = 8                # chunks handed to the LLM
RRF_K = 60               # rank-fusion damping constant
MAX_DISTANCE = 0.75      # beyond this, the corpus doesn't cover the question

REFUSAL = "I don't know — the provided HIPAA documents don't cover this."

PROMPT = """You are a HIPAA compliance assistant. Answer ONLY from the excerpts below.

RULES:
- Use no outside knowledge. If the excerpts do not answer the question, reply
  exactly: "{refusal}"
- Cite the section number after each claim, like (§ 164.404).
- Quote the regulation's own wording for requirements where possible.
- Be concise.

EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""


@dataclass
class Citation:
    id: str
    section: str
    title: str
    text: str
    distance: float | None
    fused_score: float


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    grounded: bool = True          # False when the refusal gate fired
    closest_distance: float | None = None
    llm_called: bool = True


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]{3,}", text.lower())


class RagEngine:
    """Holds the embedding model, vector store, and BM25 index as state.

    Loading these takes ~4s. In a script that's tolerable; in a server it must
    happen once at startup, not per request.
    """

    def __init__(self, use_stub_llm: bool | None = None):
        # Stub mode lets tests (Phase 3) and load tests (Phase 6) exercise
        # retrieval without consuming the Gemini free-tier quota.
        if use_stub_llm is None:
            use_stub_llm = os.getenv("CLAUSEWISE_STUB_LLM", "").lower() in ("1", "true", "yes")
        self.use_stub_llm = use_stub_llm

        self.model = SentenceTransformer(EMBED_MODEL)
        self.collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)

        # Build BM25 once over the whole corpus.
        docs = self.collection.get(include=["documents", "metadatas"])
        self.bm25_ids = docs["ids"]
        corpus = [
            _tokenize(f"{m['section']} {m['title']} {d}")
            for d, m in zip(docs["documents"], docs["metadatas"])
        ]
        self.bm25 = BM25Okapi(corpus)

        self._client = None if use_stub_llm else self._make_client()

    @staticmethod
    def _make_client():
        from google import genai
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        return genai.Client(api_key=key)

    @property
    def chunk_count(self) -> int:
        return self.collection.count()

    def retrieve(self, question: str) -> list[Citation]:
        """Hybrid retrieval: dense for paraphrase, BM25 for terms of art."""
        qv = self.model.encode([question])[0].tolist()
        dense = self.collection.query(query_embeddings=[qv], n_results=CANDIDATES)
        dense_ids = dense["ids"][0]
        dist_by_id = dict(zip(dense_ids, dense["distances"][0]))

        scores = self.bm25.get_scores(_tokenize(question))
        ranked = sorted(zip(scores, self.bm25_ids), reverse=True)
        sparse_ids = [cid for s, cid in ranked[:CANDIDATES] if s > 0]

        # Reciprocal rank fusion — uses rank position only, so the two
        # incompatible score scales never need reconciling.
        fused = defaultdict(float)
        for rank, cid in enumerate(dense_ids, 1):
            fused[cid] += 1 / (RRF_K + rank)
        for rank, cid in enumerate(sparse_ids, 1):
            fused[cid] += 1 / (RRF_K + rank)

        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K]
        got = self.collection.get(ids=[c for c, _ in top], include=["documents", "metadatas"])
        by_id = {i: (d, m) for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])}

        return [
            Citation(
                id=cid,
                section=by_id[cid][1]["section"],
                title=by_id[cid][1]["title"],
                text=by_id[cid][0],
                distance=dist_by_id.get(cid),
                fused_score=f,
            )
            for cid, f in top
        ]

    def _generate(self, question: str, citations: list[Citation]) -> str:
        if self.use_stub_llm:
            secs = ", ".join(sorted({c.section for c in citations}))
            return f"[STUB] Answer synthesized from {secs}."

        context = "\n\n".join(f"[{c.section} {c.title}]\n{c.text}" for c in citations)
        interaction = self._client.interactions.create(
            model=LLM_MODEL,
            input=PROMPT.format(refusal=REFUSAL, context=context, question=question),
        )
        return interaction.output_text.strip()

    def ask(self, question: str) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")

        citations = self.retrieve(question)
        closest = min((c.distance for c in citations if c.distance is not None), default=None)

        # Grounding gate. Deterministic, and fires BEFORE the API call — an
        # off-topic question costs zero tokens and can't be prompted around.
        if closest is None or closest > MAX_DISTANCE:
            return Answer(
                question=question,
                answer=REFUSAL,
                citations=[],
                grounded=False,
                closest_distance=closest,
                llm_called=False,
            )

        return Answer(
            question=question,
            answer=self._generate(question, citations),
            citations=citations,
            grounded=True,
            closest_distance=closest,
            llm_called=not self.use_stub_llm,
        )