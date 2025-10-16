from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition
from typing import List, Dict, Any
from config.settings import settings

class EmbeddingStore:
    def __init__(self):
        self.client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.collection_name = settings.QDRANT_COLLECTION
    
    async def initialize(self):
        """Create collection if it doesn't exist"""
        collections = await self.client.get_collections()
        
        if self.collection_name not in [c.name for c in collections.collections]:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE
                )
            )
    
    async def store_embedding(self, entity_id: str, embedding: List[float], 
                            metadata: Dict[str, Any]):
        """Store entity embedding with metadata"""
        await self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=entity_id,
                    vector=embedding,
                    payload=metadata
                )
            ]
        )
    
    async def search_similar(self, query_embedding: List[float], 
                           limit: int = 10,
                           filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar entities"""
        search_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match={"value": v})
                for k, v in filters.items()
            ]
            search_filter = Filter(must=conditions)
        
        results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=search_filter
        )
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "metadata": hit.payload
            }
            for hit in results
        ]
