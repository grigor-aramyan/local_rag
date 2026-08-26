FROM python:3.12-slim AS models
RUN pip install --no-cache-dir fastembed
ENV HF_HOME=/models
RUN python -c "\
from fastembed import TextEmbedding, TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5'); \
TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"

FROM python:3.12-slim
COPY --from=models /models /models
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ /app
ENV HF_HOME=/models \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=4
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]