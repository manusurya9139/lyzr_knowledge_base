from typing import Dict, Any, List
from openai import AsyncOpenAI
from config.settings import settings
from services.graph.graph_interface import GraphDBInterface

class QueryGeneratorTool:
    """Generate and execute Cypher/Gremlin queries"""
    
    name = "query_generator"
    description = "Generate custom graph database queries"
    
    def __init__(self, graph_db: GraphDBInterface, db_type: str = "neo4j"):
        self.graph_db = graph_db
        self.db_type = db_type
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    async def execute(self, natural_language_query: str) -> List[Dict[str, Any]]:
        """Generate and execute query from natural language"""
        query_language = "Cypher" if self.db_type == "neo4j" else "Gremlin"
        
        prompt = f"""
        Generate a {query_language} query for this request:
        {natural_language_query}
        
        Return only the query, no explanations.
        """
        
        response = await self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        query = response.choices[0].message.content.strip()
        
        # Execute query
        results = await self.graph_db.execute_query(query)
        
        return results
