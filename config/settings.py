from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    
    # OpenAI Configuration
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIMENSION: int = 3072
    
    # Neo4j Configuration
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str
    
    # AWS Neptune Configuration (optional)
    NEPTUNE_ENDPOINT: Optional[str] = None
    NEPTUNE_PORT: int = 8182
    AWS_REGION: str = "us-east-1"
    
    # Vector Store Configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "knowledge_graph"
    
    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    
    # Processing Configuration
    MAX_CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    ENTITY_RESOLUTION_THRESHOLD: float = 0.85
    DEDUP_THRESHOLD: float = 0.95
    
    class Config:
        env_file = ".env"

settings = Settings()
