from typing import List, Union, Dict, Any
from openai import AsyncOpenAI
from config.settings import settings

class EmbeddingService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    async def embed_text(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for text using OpenAI"""
        texts = [text] if isinstance(text, str) else text
        
        response = await self.client.embeddings.create(
            model=settings.OPENAI_EMBEDDING_MODEL,
            input=texts
        )
        
        embeddings = [item.embedding for item in response.data]
        return embeddings[0] if isinstance(text, str) else embeddings
    
    async def embed_entity(self, entity: Dict[str, Any]) -> List[float]:
        """Generate embedding for entity"""
        text = f"{entity['type']}: {entity['name']}"
        if 'attributes' in entity:
            attrs = ", ".join(f"{k}={v}" for k, v in entity['attributes'].items())
            text += f" ({attrs})"
        
        return await self.embed_text(text)
    
    async def embed_relationship(self, rel: Dict[str, Any]) -> List[float]:
        """Generate embedding for relationship"""
        text = f"{rel['source']} -{rel['type']}-> {rel['target']}"
        if 'properties' in rel:
            text += f" {rel['properties']}"
        
        return await self.embed_text(text)
