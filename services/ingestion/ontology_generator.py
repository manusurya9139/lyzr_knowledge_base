from typing import Dict, Any, List
from openai import AsyncOpenAI
from config.settings import settings
import json
import yaml

class OntologyGenerator:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        with open('config/prompts.yaml', 'r') as f:
            self.prompts = yaml.safe_load(f)
    
    async def generate_ontology(self, document_text: str) -> Dict[str, Any]:
        """Generate ontology from document using LLM"""
        prompt = self.prompts['ontology_generation'].format(document_text=document_text)
        
        response = await self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert ontology engineer. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        ontology = json.loads(response.choices[0].message.content)
        
        return {
            "version": "1.0",
            "entity_types": ontology.get("entity_types", []),
            "relationship_types": ontology.get("relationship_types", []),
            "hierarchies": ontology.get("hierarchies", []),
            "attributes": ontology.get("attributes", {}),
            "constraints": ontology.get("constraints", [])
        }
    
    async def refine_ontology(self, ontology: Dict[str, Any], 
                             feedback: str) -> Dict[str, Any]:
        """Refine ontology based on user feedback"""
        prompt = f"""
        Current ontology:
        {json.dumps(ontology, indent=2)}
        
        User feedback: {feedback}
        
        Refine the ontology based on the feedback. Return the complete updated ontology as JSON.
        """
        
        response = await self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    async def merge_ontologies(self, ontologies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple ontologies into unified schema"""
        prompt = f"""
        Merge these ontologies into a unified, consistent schema:
        {json.dumps(ontologies, indent=2)}
        
        Resolve conflicts, eliminate redundancies, and maintain semantic consistency.
        Return the merged ontology as JSON.
        """
        
        response = await self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
