# syntax=docker/dockerfile:1

# Stage 1: download the ONNX weights once, into an image layer of their own.
FROM python:3.12-slim AS models
# fastembed does NOT use HF_HOME for its ONNX cache — it writes to
# FASTEMBED_CACHE_PATH (default: a temp dir). Both must point at /models, or the
# weights land somewhere the runtime stage never copies and startup fails
# offline.
ENV HF_HOME=/models \
    FASTEMBED_CACHE_PATH=/models
RUN pip install --no-cache-dir fastembed==0.7.1
# TextCrossEncoder is not re-exported at the fastembed top level.
RUN python -c "\
from fastembed import TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5'); \
TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"

# Stage 2: dependencies and weights. Both land before any app code, so editing
# a source file does not invalidate several hundred MB of downloads.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    FASTEMBED_CACHE_PATH=/models \
    MODEL_CACHE_PATH=/models \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=4
WORKDIR /app
COPY --from=models /models /models
COPY requirements.txt ./
RUN pip install --no-cache-dir --retries 5 --timeout 120 -r requirements.txt

# Test image: `docker compose --profile dev run --rm tests`
FROM base AS dev
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir --retries 5 --timeout 120 -r requirements-dev.txt
COPY pyproject.toml ./
COPY app/ ./app/
COPY tests/ ./tests/
CMD ["pytest"]

FROM base AS runtime
RUN useradd --system --create-home --uid 1000 appuser \
 && mkdir -p /data/lancedb /data/documents \
 && chown -R appuser:appuser /data
COPY app/ ./app/
USER appuser
EXPOSE 8000
# One worker, deliberately: LanceDB uses optimistic concurrency over a manifest,
# so a second writer process on the same table produces commit conflicts.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
