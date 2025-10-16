from typing import List, Dict, Tuple, Optional, Any
from openai import AsyncOpenAI
from config.settings import settings
import numpy as np
from .graph_interface import GraphDBInterface

class EntityResolver:
    def __init__(self, graph_db: GraphDBInterface):
        self.graph_db = graph_db
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    async def resolve_entity(self, new_entity: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Resolve if new entity matches existing entity
        Returns: (is_duplicate, existing_entity_id)
        """
        # Find candidates by entity type
        candidates = await self.graph_db.find_nodes(
            labels=[new_entity['type']],
            properties={}
        )
        
        if not candidates:
            return False, None
        
        # Compare with candidates
        for candidate in candidates:
            similarity = await self._compare_entities(new_entity, candidate.properties)
            if similarity > settings.ENTITY_RESOLUTION_THRESHOLD:
                return True, candidate.id
        
        return False, None
    
    async def _compare_entities(self, entity1: Dict, entity2: Dict) -> float:
        """Compare two entities using LLM reasoning and embedding similarity"""
        # Embedding-based similarity
        emb1 = entity1.get('embedding')
        emb2 = entity2.get('embedding')
        
        if emb1 and emb2:
            emb_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        else:
            emb_sim = 0.0
        
        # LLM-based reasoning for complex cases
        if emb_sim > 0.7:
            prompt = f"""
            Determine if these entities refer to the same thing:
            
            Entity 1: {entity1}
            Entity 2: {entity2}
            
            Return only a number between 0.0 and 1.0 indicating similarity.
            """
            
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            
            try:
                llm_sim = float(response.choices[0].message.content.strip())
            except:
                llm_sim = 0.0
            
            # Weighted combination
            return 0.6 * emb_sim + 0.4 * llm_sim
        
        return emb_sim
    
    async def deduplicate_entities(self, entity_ids: List[str]) -> Dict[str, str]:
        """
        Find and merge duplicate entities
        Returns: mapping of duplicate_id -> canonical_id
        """
        entities = []
        for eid in entity_ids:
            nodes = await self.graph_db.find_nodes([], {"id": eid})
            if nodes:
                entities.append(nodes[0])
        
        duplicates = {}
        processed = set()
        
        for i, e1 in enumerate(entities):
            if e1.id in processed:
                continue
            
            for e2 in entities[i+1:]:
                if e2.id in processed:
                    continue
                
                similarity = await self._compare_entities(e1.properties, e2.properties)
                if similarity > settings.DEDUP_THRESHOLD:
                    duplicates[e2.id] = e1.id
                    processed.add(e2.id)
        
        return duplicates
