from typing import List, Dict, Any
from openai import AsyncOpenAI
from config.settings import settings
import json
import yaml

class EntityExtractor:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        with open('config/prompts.yaml', 'r') as f:
            self.prompts = yaml.safe_load(f)
    
    async def extract_entities(self, text: str, ontology: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entities from text according to ontology"""
        prompt = self.prompts['entity_extraction'].format(
            ontology=json.dumps(ontology),
            text=text
        )
        
        response = await self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("entities", [])
    
    async def extract_relationships(self, text: str, entities: List[Dict[str, Any]], 
                                   ontology: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract relationships between entities"""
        prompt = self.prompts['relationship_extraction'].format(
            entities=json.dumps(entities),
            ontology=json.dumps(ontology),
            text=text
        )
        
        response = await self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("relationships", [])
