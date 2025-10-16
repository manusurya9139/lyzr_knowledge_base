from typing import List, Dict, Any
from services.vector_store.embedding_store import EmbeddingStore
from services.ingestion.embedding_service import EmbeddingService

class VectorSearchTool:
    """Semantic similarity search using embeddings"""
    
    name = "vector_search"
    description = "Search for entities semantically similar to the query"
    
    def __init__(self):
        self.embedding_store = EmbeddingStore()
        self.embedding_service = EmbeddingService()
    
    async def execute(self, query: str, limit: int = 10, 
                     filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute vector similarity search"""
        # Generate query embedding
        query_embedding = await self.embedding_service.embed_text(query)
        
        # Search similar entities
        results = await self.embedding_store.search_similar(
            query_embedding,
            limit=limit,
            filters=filters
        )
        
        return results
