# ClauseWise — Build Log & Decision Record

A running record of *what* was built, *why* each choice was made, and *what it cost*.
Written for two audiences: future-me returning after a two-week gap, and an
interviewer asking "why did you do it that way?"

**Status: Phases 1–4 complete. Phase 5 mostly complete (Prometheus scraping; Grafana
dashboards pending). Phase 6 blocked on metrics-server — see §21.**

---

## Where to pick up

Everything is in version control and manifests. To rebuild from scratch:

```powershell
kind create cluster --name clausewise
docker build -t clausewise:dev .
kind load docker-image clausewise:dev --name clausewise
kubectl create secret generic clausewise-secrets --from-literal=GEMINI_API_KEY=<key>
kubectl apply -f k8s/
```

**Blocking issue:** `kubectl top pods` returns "Metrics API not available", so the HPA
reports `<unknown>` and cannot scale. See §21 for the diagnosis and options.

---

# Phase 1 — Working RAG Core

**Goal:** ask a HIPAA question in a script, get a grounded, cited answer — and a
refusal when the documents don't cover it.

```
Q: How long do I have to notify individuals after a breach?
A: A covered entity must provide the notification "without unreasonable delay and
   in no case later than 60 calendar days after discovery of a breach" (§ 164.404).

Q: What is the best pizza topping?
A: I don't know — the provided HIPAA documents don't cover this.
   (closest match 0.902 exceeded threshold 0.75 — LLM never called)
```

## What a RAG system is, and which parts exist here

**Retrieval-Augmented Generation:** rather than relying on what a language model
memorized during training, you retrieve relevant passages from your own documents and
hand them to the model as context. The model's job becomes *reading and synthesizing*,
not *recalling*. That is what makes citations possible and hallucination controllable.

| Stage | Implementation | File |
|---|---|---|
| Ingest | PDF → text, artifact removal | `clean_text.py` |
| Chunk | Section-aware splitting + citation metadata | `chunk.py` |
| Embed | Local sentence-transformers → Chroma | `embed.py` |
| Retrieve | Hybrid dense + BM25, reciprocal rank fusion | `rag.py` |
| Generate | Gemini, constrained to excerpts, cited | `rag.py` |
| Refuse | Distance threshold gate before the API call | `rag.py` |
| Serve | FastAPI, engine loaded once at startup | `api.py` |
| Display | Streamlit, with visible retrieval scores | `ui.py` |
| Test | 16 tests, stub-LLM mode | `tests/` |
| Ship | Docker + GitHub Actions → GHCR | `Dockerfile`, `ci.yml` |
| Run | Kubernetes, 2 replicas, probes | `k8s/` |
| Observe | Prometheus + custom RAG metrics | `api.py`, `servicemonitor.yaml` |

## 1. Environment

### Decision: build on `D:\ClauseWise` (secondary NTFS drive)

| Option | Verdict | Reason |
|---|---|---|
| `C:\dev\clausewise` | Fallback | Only ~31 GB free; Docker's WSL2 disk alone can reach 20 GB+ |
| OneDrive folder | **Rejected** | Sync locks files mid-write → `WinError 32` during pip installs; Files On-Demand evicts packages → phantom `ModuleNotFoundError`; races with `.git` writes. Cloud storage is not disk. |
| `D:\ClauseWise` | **Chosen** | 860 GB free, NTFS, fixed disk |

**Problem:** creating the folder through Explorer's admin prompt left it owned by
`Administrators`, so the normal account couldn't write inside it.

```powershell
takeown /F "D:\ClauseWise" /R /D Y
icacls "D:\ClauseWise" /grant "$($env:USERNAME):(OI)(CI)F" /T
icacls "D:\ClauseWise" /inheritance:e
```

**Verified NTFS before committing to the drive.** Python venvs and Chroma's SQLite
store are unreliable on exFAT/FAT32. One `Get-Volume` check avoided a class of bug
that would have surfaced much later looking like application errors.

### Dependencies and why each

| Package | Role | Rationale |
|---|---|---|
| `google-genai` | Answer generation | Free tier, no card. **Note:** Google moved to the Interactions API; older `google-generativeai` patterns no longer apply. |
| `sentence-transformers` | Embeddings | Runs **locally on CPU** — no cost, no rate limit. Deliberate: embedding runs on every query, so an API embedder would collide with free-tier quota during load testing. |
| `chromadb` | Vector store | Embedded, file-backed, zero infrastructure. |
| `rank-bm25` | Keyword retrieval | Added mid-session to fix a retrieval failure (§5). |
| `pypdf` | PDF extraction | Pure Python — matters for a slim Docker image. |
| `python-dotenv` | Secrets | Keeps the API key out of source control. |
| `fastapi` / `uvicorn` | HTTP service | Everything downstream (Docker, K8s, HPA, Prometheus) wraps a *server*. |
| `streamlit` / `requests` | Demo UI | Makes a demo video possible. |
| `pytest` / `httpx` | Tests | `httpx` backs FastAPI's `TestClient`. |
| `prometheus-fastapi-instrumentator` / `prometheus-client` | Metrics | Phase 5. |

## 2. Corpus

**Source:** HIPAA Administrative Simplification Combined Regulation Text (45 CFR Parts
160, 162, 164). 115 pages, public domain.

Every provision carries a numbered marker — `§ 164.312 Technical safeguards.` Those
numbers become **citation targets that are structural facts, not model output.**

This is the central design decision of the project. A RAG system answering "according
to the documents…" is unverifiable. One answering "per § 164.404…" can be checked
against the source in ten seconds. For a compliance tool that difference *is* the
value — and it only works because citations are captured during ingestion and never
generated by the LLM.

## 3. Ingestion

Built as separate scripts per stage rather than one pipeline, so each is independently
inspectable.

```
data/*.pdf → inspect_pdfs.py → clean_text.py → chunk.py → embed.py → chroma_db/
```

### Step 1 — `inspect_pdfs.py`: measure before designing

**Principle: never design a chunking strategy before reading the actual extracted
text.** PDF extraction is lossy in document-specific ways; blind assumptions are wrong.

- 115 pages → **470,012 characters** (confirms a real text layer; a scanned PDF would
  have shown near-zero and required OCR)
- **872 section markers** found

Three defects visible immediately:

1. **Running footer injected mid-sentence** — on all 115 pages, splitting words
   (`elec tronic`)
2. **Hard line wrapping** at ~30 chars (narrow two-column layout)
3. **Inflated marker count** — the first 8 pages are a table of contents

### Step 2 — `clean_text.py`: remove artifacts

Drop header/date/page-number lines; rejoin wrapped lines; repair hyphenation; skip TOC
pages via dot-leader density; strip Federal Register amendment citations (legislative
history — pure noise for question answering).

**Result:** 470,012 → **413,328 chars**. Footer leftovers: **0**. Word-split damage:
`electronic` 115 vs `elec tronic` 1 — negligible.

The `elec tronic` counter is a deliberate **quality probe**, not decoration. Page-break
word splits can't be fully repaired automatically, so the pipeline *measures* the
damage. Had the ratio been bad, the correct response was to change extraction strategy,
not proceed and hope.

## 4. Chunking

**Why not fixed-size chunks?** Splitting every N characters produces chunks that begin
mid-sentence and straddle two unrelated provisions. The model then answers using half
of § 164.308 and half of § 164.310 — fluent, authoritative, wrong. In a compliance tool
that is the failure mode that matters most.

### Bug 1 — cross-references matched as section headers

First run produced a chunk titled `§ 164.312 , § 164`.

**Cause:** the regex matched *"in accordance with § 164.312, § 164.314 and § 164.316"*
inside the body of § 164.306. The pattern couldn't distinguish a section being
**defined** from one being **mentioned**.

**Impact if shipped:** § 164.306's text labeled as § 164.312 — a confidently mislabeled
citation.

**Fix:** require a Title-Case title ending in a period, followed by a subsection marker
`(a)` or a new sentence. Cross-references fail this test.

```python
SECTION_RE = re.compile(
    r"§\s*(\d{3}\.\d{3,4})\s+"          # § 164.312
    r"([A-Z][^.§]{3,140}?)\."           # Title Case, ending in a period
    r"(?=\s+(?:\([a-z0-9]\)|[A-Z]))"    # real body follows
)
```

Suspicious labels: **128 → 5.**

### Bug 2 — duplicate section IDs

Chroma rejected the insert: 13 duplicated IDs. Some section numbers matched twice —
once in a subpart contents listing, once as the real provision.

**Fix:** when a number matches more than once, keep the occurrence with the most body
text (contents entries are short; provisions are long). Then `assert` ID uniqueness so
this can never reach the vector store again.

**Why assert rather than deduplicate silently:** silent data-quality failures are what
make RAG systems mysteriously bad. Better to crash at build time than serve wrong
citations at query time.

### Output

498 chunks · avg 953 chars · Parts 160, 162, 164 all covered · IDs unique

**Chunk size:** started at 2,500 chars, reduced to 1,200. Larger chunks average too
much unrelated content into a single vector, diluting specific provisions. Halving the
size measurably improved the breach-notification query (0.393 → 0.321) and surfaced the
exact answering sentence as the top hit.

## 5. Retrieval — the hardest part of the project

### What embeddings are

A model converts a passage into a fixed-length vector positioned so that **passages
with similar meaning land near each other.** "Notify people about a breach" and
"provide notification following discovery of a breach of unsecured protected health
information" share few words but sit close in vector space.

**Model: `all-MiniLM-L6-v2`** — 384 dimensions, ~90 MB, CPU, free. Rejected: Gemini's
embedding API (would burn free-tier quota on every query, colliding with load testing)
and larger local models (3× slower, 5× bigger, unjustified for 498 chunks).

**Enrichment:** section number and title are prepended to the text *before* embedding,
but the clean body is what's stored for the LLM to read. The heading is the densest
semantic signal in a chunk; leaving it in metadata only discards it from the vector.

**Metric:** cosine, set explicitly. Chroma defaults to squared L2. Sentence-transformer
vectors are normalized, so cosine is correct — with L2 nothing errors, results are just
quietly worse.

### Measured retrieval quality *before* building generation

| Query | Result |
|---|---|
| Breach notification timing | **Excellent** — § 164.404, distance 0.321, exact answering sentence |
| Business associate definition | **Good** — correct definition retrieved |
| Off-topic control ("pizza") | **Excellent** — 0.90+, clean separation |
| Encryption at rest | **Failed** — § 164.312 not in top 40 of 498 |

Had generation been built first, the encryption query would have produced a confident,
well-cited, **wrong** answer — and the cause would have been invisible behind fluent
prose.

### Bug 3 — substring collisions in keyword matching

The first hybrid-search attempt scored chunks by counting query terms via a 6-character
prefix match. `rest` matched `interest`, `restriction`, `requested`; `data` matched
almost everything. The keyword retriever promoted noise and rank fusion amplified it.
**Results got worse, and a previously-working query regressed.**

**Fix: BM25** (`rank-bm25`) — down-weights terms appearing in many documents (so `data`
counts for almost nothing while `encrypt` counts heavily), normalizes for document
length, matches whole tokens rather than substrings. Restored the regressed query
immediately.

### Reciprocal Rank Fusion

Dense and sparse retrievers produce scores on incompatible scales, so any weighted
blend needs a magic constant requiring endless tuning. RRF uses only **rank position**:

```
score(doc) = Σ  1 / (k + rank_in_that_retriever)
```

A chunk both retrievers rank highly wins, regardless of scale. Six lines, standard
practice, no tuning.

### Bug 4 — vocabulary mismatch (unresolved, documented)

§ 164.312 contains the literal answer to the encryption query:

> "Encryption and decryption (Addressable). Implement a mechanism to encrypt and
> decrypt electronic protected health information."

The chunk is correctly extracted, correctly labeled, present in the index — **and ranks
below 40th of 498.**

**Diagnosed, not guessed:** wrote `check_312.py` to confirm the chunk exists, verify its
label, list every chunk containing "encrypt" (6 of 498), and locate the target's actual
rank. Only then was the cause identifiable.

**Root cause:** `all-MiniLM-L6-v2` is a general-purpose model. It does not encode that
"patient data at rest" and "electronic protected health information" are the same
concept. Meanwhile § 164.514 discusses patients and data constantly, winning on surface
similarity while being about an entirely different topic.

| Attempt | Outcome |
|---|---|
| Reduce chunk size 2500 → 1200 | No effect on this query (helped others) |
| Prepend title/section to embedded text | No effect on this query (helped others) |
| Hybrid BM25 + dense with RRF | Fixed a *different* regression; not this |

**Deferred fix:** domain-adapted embedding model, or LLM-based query expansion.
**Why deferred:** four iterations spent; the gap is a model-capability limit, not a bug.
Phase 7 builds retrieval-quality monitoring — the correct place to measure this against
real query distributions rather than four hand-picked questions.

## 6. Generation and grounding

### Two-layer anti-hallucination

**Layer 1 — distance threshold gate, before the API call.** If the closest chunk
exceeds 0.75 cosine distance, return the refusal without invoking the LLM at all. It
cannot be talked out of refusing by clever phrasing, and an off-topic query **costs
zero tokens**.

**Layer 2 — prompt constraint.** Instructs refusal when excerpts are insufficient.

Defense in depth, because prompt rules alone are not reliable. The threshold is
deterministic; the prompt is probabilistic.

**Threshold calibration:** measured, not guessed. On-topic queries scored 0.32–0.46;
the off-topic control scored 0.90+. 0.75 sits in the gap. This is why an off-topic probe
query existed in the test set from the very first retrieval run.

---

# Phase 2 — API + UI

## 7. Why Phase 2 exists at all

Every phase after this wraps around an HTTP service. Docker containerizes a **server**.
Kubernetes schedules a **server**. The HPA scales a **server** by request load.
Prometheus scrapes a **server's** metrics endpoint.

Containerizing `ask.py` would have produced a container that runs once and exits.

## 8. The `RagEngine` refactor

**The problem, visible in the Phase 1 logs:** three questions produced three separate
model loads. `ask.py` reloaded the 90 MB embedding model and rebuilt the BM25 index over
all 498 chunks on *every invocation* — roughly 4 seconds each.

Tolerable in a script. Fatal in a server, and it would have made the autoscaling demo
meaningless: Grafana would have been measuring model loading, not request handling.

**The fix:** a `RagEngine` class holding the embedding model, Chroma collection, and
BM25 index as instance state. `ask.py` (CLI) and `api.py` (HTTP) both became thin
wrappers over the same engine. `ask.py` shrank to ~20 lines.

### The stub-LLM switch

`CLAUSEWISE_STUB_LLM=1` replaces the Gemini call with canned text while leaving
retrieval fully intact. Two payoffs, both planned from day one:

- **CI** — tests exercise retrieval end-to-end with no API key and no quota
- **Load testing** — thousands of synthetic requests without hitting the 10–15
  requests/minute free-tier ceiling

This is why local embeddings were chosen in Phase 1: only *generation* is rate limited,
so isolating it behind one swappable method makes the whole system load-testable.

**This paid off unexpectedly in Phase 5** — when the Gemini key stopped working, stub
mode let the Kubernetes work continue with zero LLM dependency (§20).

## 9. Bug 5 — relative paths break when the working directory changes

`DB_DIR = "chroma_db"` resolved relative to wherever the process was launched. Running
uvicorn from `src/` made Chroma fail with `Collection [hipaa] does not exist`.

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = str(PROJECT_ROOT / "chroma_db")
```

`__file__` doesn't change with the working directory. Docker sets its own `WORKDIR` and
Kubernetes sets another; this makes both irrelevant. Caught here for five minutes of
work rather than inside a container.

## 10. The API

**`POST /ask`** → answer, citations, `grounded`, `llm_called`, `closest_distance`.
**`GET /health`** → status + chunk count; used as the Kubernetes startup, readiness, and
liveness probe.

**Startup via FastAPI's `lifespan` hook** — the engine loads once before the server
accepts traffic. Visible in the logs: requests produce no model reloading.

**Validation is declarative.** `Field(min_length=3, max_length=1000)` rejects bad input
with a 422 *before* application code runs, and generates the `/docs` interface for free.

**Error mapping** — Gemini rate limits become a clean 429 rather than a 500 and a stack
trace.

### Observed: both refusal layers fire in production

A live query retrieved excerpts *within* threshold (closest 0.533) but the model
correctly declined to answer from them. The two layers catch genuinely different failure
modes:

- **Layer 1** (distance gate) — questions outside the corpus entirely
- **Layer 2** (prompt constraint) — related text exists but isn't responsive

The UI now distinguishes them, because for a compliance user "we found nothing relevant"
and "we found related text that doesn't answer you" are different answers.

## 11. Observed: RAG handles lookup, not aggregation

The query *"what is this document about"* retrieved eight mediocre matches (0.53–0.67)
and the model declined.

**Root cause is architectural, not a defect.** Retrieval selects passages by similarity
to the question. A question with no specific subject has nothing to match against. The
same applies to *"summarize the Privacy Rule"* — these need the whole corpus at once,
which a retrieval system by definition never sees.

**Addressed by scoping guidance in the UI**, not by engineering around it. A hardcoded
summary path would be a canned answer disguised as retrieval, and for a compliance tool,
overstating capability is a liability.

## 12. The UI

- **Retrieval distances shown next to every source.** Most RAG demos hide this. Exposing
  it makes grounding *auditable*. That is a compliance instinct, not a chatbot one.
- **Sidebar health check** — degrades gracefully when the API is down.
- `API_URL` is a single constant, so pointing it at a Kubernetes service DNS name is a
  one-line change.

---

# Phase 3 — Containerize + CI/CD

## 13. The bake-vs-build decision

**Question:** bake `chroma_db/` into the image, or build it at container startup?

**Chose: bake it in.** The index is a *build artifact*, deterministic output from
`chunks.json` — not runtime state. Baking it in gives an **immutable image**: the
container that passes CI is byte-identical to the one that runs in production.

**The decider was autoscaling.** Under load the HPA spins up new pods. A 30-second index
build on every pod start means new capacity arrives *after* the traffic spike has
passed — the autoscaling demo would show pods scaling up while latency stays terrible.

The cost is image size, but the index is ~5 MB against a PyTorch base of ~700 MB. Noise.

## 14. Image size: 703 MB

The real size problem was PyTorch, not the index.

**The default `torch` wheel bundles CUDA runtime libraries** — roughly 2 GB of GPU
support that is dead weight in a CPU-only container. Installing from the CPU index cut
the image about 70%:

```dockerfile
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
```

Other deliberate Dockerfile choices:

- **`requirements.txt` copied before `src/`** — layer caching. Editing a Python file
  rebuilds only the last few layers instead of re-downloading dependencies.
- **Embedding model baked in at build time.** Without it every pod hits HuggingFace on
  boot — slow, and a hard runtime dependency on an external service.
- **Non-root user.** Kubernetes pod security standards commonly reject root containers.
- **`--host 0.0.0.0`.** Binding to localhost inside a container means "reachable only
  from inside this container."
- **`.dockerignore`** — without it Docker ships the entire 2.5 GB `.venv` as build
  context, and `.env` could land inside the image.

## 15. Bug 6 — dependency drift between venv and container

The first container run died with `ModuleNotFoundError: No module named 'rank_bm25'`.

**Cause:** `rank-bm25` was `pip install`ed locally mid-session but never added to
`requirements.txt`. The local venv had it; the container — a clean environment that only
knows what was declared — did not.

This is exactly the drift the venv was supposed to prevent, and it happened anyway
because *installing* and *declaring* were separate acts. **Habit adopted:** add to
`requirements.txt` in the same motion as `pip install`. (It recurred once more in
Phase 5 with `prometheus-client`.)

## 16. The test suite

16 tests, all running with `use_stub_llm=True` — **no Gemini key, no quota, no network.**
That is what makes CI possible on a fresh clone.

Four tests assert **citation integrity** specifically. Tests that encode the worst
failure mode are the ones worth writing: shipping a wrong regulation to a compliance user
is the outcome this whole project is designed to prevent.

### Refining the citation defect measurement

The integrity test initially failed, flagging **6 of 498 chunks (1.2%)**.

Rather than accept that number, `check_labels.py` printed what each flagged chunk
actually contained. Five were **truncated but correctly attributed** — the header was
lost at a page boundary, but the body text genuinely belongs to the cited section
(`160.404-0` starts at `(b)`, mid-provision, with § 160.404's own penalty text).

Only **`164.318-0`** is genuinely mislabeled: it contains *Appendix A to Subpart C —
Security Standards: Matrix* while claiming § 164.318 (compliance dates).

**True citation error rate: 1 chunk in 498 (0.2%).**

**The test now pins the bound at 6** rather than asserting zero:

```python
assert len(bad) <= 6, f"Mislabeled count increased to {len(bad)}"
```

**Why bounded rather than `xfail`:** `xfail` hides the problem. A bounded assertion
converts a known limitation into a **regression guard** — if a future chunker change
breaks 20 chunks, CI catches it; if the cleaner gets fixed and it drops to 0, the test
still passes and the bound tightens. A test that fails on every run gets ignored, and
then it catches nothing.

## 17. CI/CD

GitHub Actions, two jobs:

1. **test** — install, build the index, run pytest with `CLAUSEWISE_STUB_LLM=1`
2. **build** — `needs: test`, so **the image only publishes if all 16 tests pass**;
   builds and pushes to GitHub Container Registry

**The index is rebuilt in CI rather than committed.** `chroma_db/` stays gitignored, and
the workflow regenerates it from `chunks.json`. This proves the index is reproducible on
a clean machine — a stronger claim than committing an unverifiable binary blob.

---

# Phase 4 — Kubernetes

## 18. Cluster and image loading

Local cluster via **kind** — the "cluster" is a Docker container running a control
plane. Free, and behaves like the real thing for everything here.

**Decision: `kind load docker-image` rather than pulling from GHCR.** The GHCR package is
private by default, so pulling would require an image pull secret with a personal access
token. `kind load` copies the image straight from the local Docker daemon into the
cluster. Faster iteration, one less moving part. GHCR pull auth is a Phase 8 concern when
deploying to a real cloud cluster.

This requires `imagePullPolicy: IfNotPresent` in the deployment — without it Kubernetes
tries Docker Hub and fails, because the image exists only inside the kind node.

## 19. Deployment design

**Three probes, each with a distinct job:**

| Probe | Purpose | Failure behavior |
|---|---|---|
| `startupProbe` | Gives the engine ~150s to load before other probes count against it | Restarts if never ready |
| `readinessProbe` | Gates traffic | Removed from Service endpoints, **not** restarted |
| `livenessProbe` | Detects a genuinely hung process | Restarted |

All three hit `GET /health`, which was built in Phase 2 for exactly this. The Docker
`HEALTHCHECK` had already proven the endpoint worked before Kubernetes ever saw it.

**Resource requests are not decoration.** `cpu: 250m` is what the HPA reads to compute
utilization — a target of 60% means 60% *of the request*, not of a core. Without a
request, autoscaling has no baseline and reports `<unknown>`.

**Secrets via `secretKeyRef`.** The manifest contains a *reference* to a secret name and
key, never a value. The secret itself is created with `kubectl create secret` and never
touches a file. This is what makes `k8s/deployment.yaml` safe to commit publicly.

**Verified self-healing:** deleting a pod causes the Deployment to schedule a replacement
within seconds while the surviving replica continues serving. `/health` returns 200
throughout. This is the single best thing to screen-record for a demo.

---

# Phase 5 — Observability

## 20. Instrumentation

`prometheus-fastapi-instrumentator` provides the standard RED metrics (rate, errors,
duration) automatically. Three custom metrics were added because **infrastructure metrics
cannot tell you whether the answers are any good:**

| Metric | Signal |
|---|---|
| `clausewise_refusals_total{layer}` | Which refusal layer is catching queries. A shift toward `distance_gate` means users are asking things the corpus doesn't cover. |
| `clausewise_retrieval_distance` | Histogram of closest-chunk distance per query. **This is the Phase 7 drift signal** — a rightward shift over time means incoming questions are moving away from the corpus. |
| `clausewise_llm_calls_total` | LLM calls actually made vs. total requests — cost-per-query, and a direct measure of how often the refusal gate saves a token spend. |

### ServiceMonitor: the label that everyone gets wrong

`kube-prometheus-stack` only discovers ServiceMonitors carrying its release label:

```yaml
metadata:
  labels:
    release: monitoring
```

Without it the target silently never appears and **nothing errors**. Worth knowing
because the failure mode is invisible.

**Result: `serviceMonitor/default/clausewise-api/0 — 2/2 up`, 5ms scrapes.** Prometheus
discovered both pods independently, so any new pod an autoscaler creates is picked up
automatically with no config change.

## 21. Bug 7 — the Gemini `AQ.` key format

Requests started failing with `400 API key not valid`, then
`401 ACCESS_TOKEN_TYPE_UNSUPPORTED`.

**Initial diagnosis was wrong.** The key in the Kubernetes secret started with `AQ.` and
was 53 characters, not the familiar 39-character `AIza...` format, so it looked like the
wrong credential had been copied.

**Actual cause:** Google AI Studio began issuing Gemini API keys with an `AQ.` prefix
alongside the legacy `AIzaSy` format. Multiple developer reports describe these keys
failing against the standard REST endpoint, and some accounts can only generate the new
format. This is an upstream issue, not a configuration error.

**Two things worth recording:**

1. **Verified graceful degradation.** With a completely invalid upstream credential, pods
   stayed `Running`, health probes kept passing, Prometheus kept scraping, and the API
   returned structured 502s rather than crashing. The Phase 2 error mapping did its job
   under a failure that was never planned for.
2. **Stub mode unblocked the work.** Setting `CLAUSEWISE_STUB_LLM=1` in the deployment
   let all Kubernetes and Prometheus work continue with real retrieval and zero LLM
   dependency. This was designed in Phase 1 for load testing; it turned out to be a
   general resilience mechanism.

## 22. Blocked: metrics-server on kind

**Status: unresolved. This blocks the HPA and therefore Phase 6.**

`kubectl top pods` returns `error: Metrics API not available`. The metrics-server
Deployment rolls out successfully, but the metrics API never becomes available.

**Likely cause:** kind nodes use self-signed kubelet certificates, so metrics-server must
run with `--kubelet-insecure-tls`. The patch command was issued, but PowerShell's handling
of the JSON escaping in `kubectl patch --type=json` is unreliable and the flag may never
have applied.

**Next steps to try, in order:**

1. Confirm the flag landed:
   `kubectl get deployment metrics-server -n kube-system -o jsonpath='{.spec.template.spec.containers[0].args}'`
2. If absent, add it via `kubectl edit deployment metrics-server -n kube-system` rather
   than a patch command — interactive editing avoids the shell-escaping problem entirely
3. Read the actual failure: `kubectl logs -n kube-system deployment/metrics-server`
   — look for `x509: cannot validate certificate`
4. Confirm registration: `kubectl get apiservice v1beta1.metrics.k8s.io` should show
   `AVAILABLE: True`
5. Allow ~60s after any change for a scrape cycle before testing

**Alternatives if it stays stubborn:**

- **prometheus-adapter** — scale on custom Prometheus metrics (request rate) instead of
  CPU. Arguably more impressive than CPU-based scaling, but ~30 minutes more setup. The
  Prometheus side already works.
- **Manual scaling demo** — `kubectl scale deployment clausewise-api --replicas=6` during
  a load test still demonstrates Grafana responding and load distributing across pods.
  Less impressive than true autoscaling, but honest and quick.

---

## 23. Known limitations

Stated deliberately. Found by measurement, not assumed away.

**Page-spanning section headers are lost.** Six of 498 chunks affected; five truncated
but correctly attributed, one (`164.318-0`) genuinely mislabeled. True error rate 0.2%.
Root cause is in `clean_text.py`, which collapses the document to a single string and
discards the line structure needed to reconstruct split headers. Pinned by a regression
test. **Fix:** preserve line boundaries through cleaning.

**Vocabulary mismatch on some queries.** See Bug 4. Deferred to Phase 7 instrumentation.

**No aggregation or summarization.** See §11. Architectural; disclosed in the UI.

**No autoscaling yet.** See §22.

---

## Interview questions this work prepares for

**"Why chunk on section boundaries instead of fixed size?"**
Fixed-size chunks straddle unrelated provisions; the model blends two regulations into
one confident, wrong answer. Section boundaries make each chunk self-contained *and*
attach a citation label for free.

**"How do you know your citations are accurate?"**
They come from document structure captured at ingestion, never generated by the LLM, and
are validated programmatically — which is how the cross-reference mislabeling bug was
caught before it shipped. Current error rate is 1 in 498, measured, with a regression
test pinning it.

**"Why hybrid retrieval?"**
Legal text is dense with terms of art where exact matching beats semantics —
"encryption" appears in 6 of 498 chunks. Dense handles paraphrase; BM25 handles precise
terminology. RRF merges them without requiring comparable scales.

**"How do you prevent hallucination?"**
Two layers: a deterministic distance gate that refuses before the LLM is called, plus a
prompt constraint. Both have been observed firing on different failure modes.

**"Why did you refactor into a class?"**
The scripts reloaded a 90 MB model on every invocation. A server must load once at
startup — otherwise autoscaling metrics would measure model loading rather than request
handling.

**"How do you load test something with a rate-limited LLM?"**
Generation sits behind one swappable method with a stub mode. Retrieval stays real; the
API call is replaced. Designed in Phase 1 when local embeddings were chosen specifically
so generation would be the only rate-limited component. It also turned out to be the
thing that kept the project moving when the upstream API key broke.

**"How did you get the image down to 703 MB?"**
The default PyTorch wheel bundles CUDA runtime libraries — dead weight without a GPU.
Installing from the CPU index cut roughly 70%. Layer ordering keeps dependency layers
cached separately from source.

**"What metrics would you monitor for a RAG system specifically?"**
Request rate and latency tell you nothing about answer quality. Refusal rate by layer,
and the distribution of retrieval distances, are the signals that actually indicate
whether the system is still serving the questions users are asking.

**"What's broken in your system?"**
One genuinely mislabeled chunk in 498 (0.2%), root-caused to the cleaning stage and
pinned by a regression test; a vocabulary-mismatch retrieval gap, root-caused to
embedding model capability with three fixes attempted and honestly reported as
ineffective; no aggregation support, which is architectural and disclosed in the UI; and
autoscaling is not yet working because metrics-server won't register on kind.

**"What would you do differently?"**
Preserve line structure through cleaning. Reach for BM25 before writing a naive keyword
matcher — the first version had substring collisions that made results actively worse.
And add packages to `requirements.txt` at install time, not at container-failure time.

---

## Next

**Immediate:** resolve metrics-server (§22), then the HPA and k6 load test.

**Then:** Grafana dashboards — request rate, p95 latency, refusals by layer, retrieval
distance p95, and pod count. The pod-count and p95-latency panels side by side *are* the
autoscaling story: latency climbs, pods scale, latency recovers.

**Learning gaps to close first:** Python OOP (classes, `self`, decorators) and Kubernetes
fundamentals (pods, deployments, services, probes) before returning to HPA work.
Debugging metrics-server is a poor place to learn what a HorizontalPodAutoscaler is.