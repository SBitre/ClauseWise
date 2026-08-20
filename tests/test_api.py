"""API contract tests."""

import os

import pytest
from fastapi.testclient import TestClient

os.environ["CLAUSEWISE_STUB_LLM"] = "1"

from src.api import app


@pytest.fixture(scope="session")
def client():
    # The context manager triggers FastAPI's lifespan hook, loading the engine.
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ask_returns_expected_shape(client):
    r = client.post("/ask", json={"question": "How long do I have to notify individuals after a breach?"})
    assert r.status_code == 200
    body = r.json()
    for key in ("question", "answer", "grounded", "llm_called", "closest_distance", "citations"):
        assert key in body


def test_short_question_rejected(client):
    assert client.post("/ask", json={"question": "ab"}).status_code == 422


def test_long_question_rejected(client):
    assert client.post("/ask", json={"question": "x" * 2000}).status_code == 422


def test_off_topic_refuses(client):
    r = client.post("/ask", json={"question": "What is the best pizza topping?"})
    assert r.json()["grounded"] is False
    assert r.json()["llm_called"] is False