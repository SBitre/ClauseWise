"""FastAPI service wrapping the RAG engine."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag import RagEngine

# Module-level holder. The engine is expensive to build (~4s), so it is created
# once at startup and reused by every request.
engine: RagEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once before the server accepts traffic, and once on shutdown."""
    global engine
    print("Loading RAG engine ...")
    engine = RagEngine()
    print(f"Ready. {engine.chunk_count} chunks indexed.")
    yield                      # server runs here
    print("Shutting down.")


app = FastAPI(
    title="ClauseWise",
    description="Grounded question answering over HIPAA regulations.",
    version="0.1.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class CitationOut(BaseModel):
    section: str
    title: str
    text: str
    distance: float | None
    fused_score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    grounded: bool
    llm_called: bool
    closest_distance: float | None
    citations: list[CitationOut]


@app.get("/health")
def health():
    """Kubernetes uses this in Phase 4 to decide when a pod is ready."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")
    return {"status": "ok", "chunks": engine.chunk_count}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")

    try:
        result = engine.ask(req.question)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        # Gemini rate limits (429) and timeouts land here. A service returns a
        # clean error and stays up; it does not crash with a stack trace.
        msg = str(e)
        if "429" in msg or "quota" in msg.lower() or "rate" in msg.lower():
            raise HTTPException(
                status_code=429,
                detail="LLM rate limit reached. Please retry in a moment.",
            )
        raise HTTPException(status_code=502, detail=f"Generation failed: {msg}")

    return AskResponse(
        question=result.question,
        answer=result.answer,
        grounded=result.grounded,
        llm_called=result.llm_called,
        closest_distance=result.closest_distance,
        citations=[
            CitationOut(
                section=c.section,
                title=c.title,
                text=c.text,
                distance=c.distance,
                fused_score=c.fused_score,
            )
            for c in result.citations
        ],
    )