from typing import List, Dict, Any
from openai import AsyncOpenAI
from config.settings import settings
import json
import yaml

class QueryPlanner:
    """Plan multi-step retrieval strategies"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        with open('config/prompts.yaml', 'r') as f:
            self.prompts = yaml.safe_load(f)
    
    async def plan_query(self, query: str) -> List[Dict[str, Any]]:
        """Generate query execution plan"""
        prompt = self.prompts['query_planning'].format(query=query)
        
        response = await self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        plan = json.loads(response.choices[0].message.content)
        return plan.get("steps", [])
    
    async def refine_plan(self, query: str, previous_results: List[Dict[str, Any]], 
                         plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Refine plan based on intermediate results"""
        prompt = f"""
        Original query: {query}
        Current plan: {json.dumps(plan)}
        Results so far: {json.dumps(previous_results)}
        
        Should we continue with the plan or adjust it? Return updated plan as JSON.
        """
        
        response = await self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        refined = json.loads(response.choices[0].message.content)
        return refined.get("steps", plan)
