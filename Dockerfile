FROM python:3.12-slim

WORKDIR /app

# Dependencies first, in their own layer. Docker caches layers, so editing
# source code won't re-download 2.5 GB of PyTorch on every rebuild.
COPY requirements.txt .

# CPU-only torch. The default wheel bundles CUDA libraries (~2 GB) that are
# useless in a container with no GPU.
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so containers don't download it at
# startup. Without this, every pod hits HuggingFace on boot.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Source and the prebuilt index. The index is a deterministic build artifact,
# so baking it in keeps the image immutable and pod startup fast.
COPY src/ ./src/
COPY chroma_db/ ./chroma_db/

# Non-root user. Kubernetes security policies commonly reject root containers.
RUN useradd -m -u 1000 clausewise && chown -R clausewise:clausewise /app
USER clausewise

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]