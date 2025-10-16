from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from .routes import documents, ontology, retrieval, health
from services.graph.neo4j_adapter import Neo4jAdapter
from services.vector_store.embedding_store import EmbeddingStore
from config.settings import settings

# Global state
graph_db = None
embedding_store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global graph_db, embedding_store
    graph_db = Neo4jAdapter()
    await graph_db.connect()
    
    embedding_store = EmbeddingStore()
    await embedding_store.initialize()
    
    yield
    
    # Shutdown
    await graph_db.disconnect()

app = FastAPI(
    title="Knowledge Graph Platform",
    description="Production-grade agentic retrieval system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(ontology.router, prefix="/ontology", tags=["ontology"])
app.include_router(retrieval.router, prefix="/retrieval", tags=["retrieval"])
