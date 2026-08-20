"""Tests for the RAG engine. Runs without a Gemini key via stub mode."""

import json
from pathlib import Path

import pytest

from src.rag import RagEngine, REFUSAL, MAX_DISTANCE

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def engine():
    """Built once for the whole session — loading takes ~4s."""
    return RagEngine(use_stub_llm=True)


@pytest.fixture(scope="session")
def chunks():
    path = PROJECT_ROOT / "data/processed/chunks.json"
    return json.loads(path.read_text(encoding="utf-8"))


# --- Data integrity -------------------------------------------------------

def test_chunk_ids_are_unique(chunks):
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_all_parts_present(chunks):
    assert {c["part"] for c in chunks} == {"160", "162", "164"}


def test_titles_are_not_cross_references(chunks):
    """A title containing § means the regex matched a cross-reference rather than
    a section definition — the bug that caused mislabeled citations."""
    bad = [c for c in chunks if "§" in c["title"]]
    assert not bad, f"Cross-reference titles: {[c['id'] for c in bad][:5]}"


def test_citation_label_integrity_does_not_regress(chunks):
    """Known limitation: 6 chunks lose their section header where it falls across
    a PDF page boundary (see BUILD_LOG.md §13). Five are truncated but correctly
    attributed; one (164.318-0) contains Appendix A content and is genuinely
    mislabeled. This test pins the count so it can only go down, never up.
    """
    bad = [
        c for c in chunks
        if c["id"].endswith("-0")
        and not c["text"].lstrip().startswith(f"§ {c['id'].rsplit('-', 1)[0]}")
    ]
    assert len(bad) <= 6, (
        f"Mislabeled count increased to {len(bad)}: {[c['id'] for c in bad]}"
    )


# --- Retrieval ------------------------------------------------------------

def test_index_loaded(engine):
    assert engine.chunk_count > 400


def test_retrieval_returns_results(engine):
    hits = engine.retrieve("breach notification timing")
    assert len(hits) > 0
    assert all(h.section.startswith("§") for h in hits)


def test_breach_query_finds_correct_section(engine):
    hits = engine.retrieve("How long do I have to notify individuals after a breach?")
    sections = {h.section for h in hits}
    assert "§ 164.404" in sections


def test_off_topic_query_is_distant(engine):
    hits = engine.retrieve("What is the best pizza topping?")
    closest = min(h.distance for h in hits if h.distance is not None)
    assert closest > MAX_DISTANCE


# --- Grounding gate -------------------------------------------------------

def test_off_topic_refuses_without_calling_llm(engine):
    result = engine.ask("What is the best pizza topping?")
    assert result.answer == REFUSAL
    assert result.grounded is False
    assert result.llm_called is False
    assert result.citations == []


def test_on_topic_passes_the_gate(engine):
    result = engine.ask("How long do I have to notify individuals after a breach?")
    assert result.grounded is True
    assert len(result.citations) > 0
    assert result.closest_distance < MAX_DISTANCE


def test_empty_question_rejected(engine):
    with pytest.raises(ValueError):
        engine.ask("   ")