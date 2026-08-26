from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
    app.state.reranker = TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
    app.state.db = lancedb.connect(os.environ["LANCEDB_PATH"])
    yield

app = FastAPI(lifespan=lifespan)