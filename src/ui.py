"""Streamlit UI for ClauseWise."""

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="ClauseWise", page_icon="§", layout="centered")

st.title("ClauseWise")
st.caption("Grounded question answering over the HIPAA regulations (45 CFR 160, 162, 164)")

# Sidebar: service status. Confirms the API is reachable before the user asks.
with st.sidebar:
    st.subheader("Service")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"Connected · {health['chunks']} chunks indexed")
    except Exception:
        st.error("API unreachable")
        st.code("uvicorn src.api:app --port 8000")

    st.markdown("---")
    st.caption(
        "Answers are generated only from retrieved excerpts. "
        "If nothing in the corpus is close enough, the system refuses "
        "rather than guessing."
    )

question = st.text_input(
    "Ask a compliance question",
    placeholder="How long do I have to notify individuals after a breach?",
)

if st.button("Ask", type="primary") and question:
    with st.spinner("Retrieving and generating..."):
        try:
            resp = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=60)
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the API: {e}")
            st.stop()

    if resp.status_code == 422:
        st.warning("Question must be between 3 and 1000 characters.")
        st.stop()
    if resp.status_code == 429:
        st.warning("Rate limit reached. Try again in a moment.")
        st.stop()
    if resp.status_code != 200:
        st.error(f"Error {resp.status_code}: {resp.text}")
        st.stop()

    data = resp.json()

    # Two distinct refusal paths, and a compliance user needs to tell them apart:
    #   Layer 1 (grounded=False) — nothing in the corpus was close enough; LLM never called.
    #   Layer 2 (grounded=True)  — excerpts retrieved, but the model judged them
    #                              insufficient to answer.
    refused = data["answer"].startswith("I don't know")

    if refused:
        st.info(data["answer"])
        if not data["grounded"]:
            st.caption(
                f"Retrieval gate: closest match {data['closest_distance']:.3f} exceeded "
                f"the 0.75 threshold — the language model was never called."
            )
        else:
            st.caption(
                f"Excerpts were retrieved (closest {data['closest_distance']:.3f}) but the "
                f"model judged them insufficient to answer. Sources shown below for review."
            )
    else:
        st.markdown("### Answer")
        st.write(data["answer"])

    if data["citations"]:
        st.markdown("### Sources")
        st.caption("Every claim above is drawn only from these excerpts.")
        for c in data["citations"]:
            dist = f"{c['distance']:.3f}" if c["distance"] is not None else "keyword"
            with st.expander(f"{c['section']} — {c['title']}   ·   distance {dist}"):
                st.write(c["text"])