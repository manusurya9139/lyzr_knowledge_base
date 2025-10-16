from typing import List, Dict, Any, AsyncGenerator
from .query_planner import QueryPlanner
from .tools.vector_search import VectorSearchTool
from .tools.graph_traversal import GraphTraversalTool
from .tools.logical_filter import LogicalFilterTool
from .tools.query_generator import QueryGeneratorTool
from services.graph.graph_interface import GraphDBInterface
from openai import AsyncOpenAI
from config.settings import settings
import json

class AgentOrchestrator:
    """Autonomous agent for intelligent retrieval"""
    
    def __init__(self, graph_db: GraphDBInterface, db_type: str = "neo4j"):
        self.planner = QueryPlanner()
        self.tools = {
            "vector_search": VectorSearchTool(),
            "graph_traversal": GraphTraversalTool(graph_db),
            "logical_filter": LogicalFilterTool(graph_db),
            "query_generator": QueryGeneratorTool(graph_db, db_type)
        }
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    async def retrieve(self, query: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute retrieval with streaming responses"""
        # Step 1: Plan query
        plan = await self.planner.plan_query(query)
        yield {"type": "plan", "content": plan}
        
        # Step 2: Execute plan
        all_results = []
        
        for step_idx, step in enumerate(plan):
            tool_name = step["tool"]
            params = step["params"]
            reasoning = step.get("reasoning", "")
            
            yield {
                "type": "step",
                "step": step_idx + 1,
                "tool": tool_name,
                "reasoning": reasoning
            }
            
            # Execute tool
            if tool_name in self.tools:
                results = await self.tools[tool_name].execute(**params)
                all_results.extend(results)
                
                yield {
                    "type": "step_result",
                    "step": step_idx + 1,
                    "results": results
                }
            
            # Check if we need to refine the plan
            if step_idx < len(plan) - 1:
                refined_plan = await self.planner.refine_plan(query, all_results, plan[step_idx+1:])
                if refined_plan != plan[step_idx+1:]:
                    plan = plan[:step_idx+1] + refined_plan
                    yield {"type": "plan_refined", "new_plan": refined_plan}
        
        # Step 3: Synthesize final answer
        final_answer = await self._synthesize_answer(query, all_results)
        yield {"type": "final_answer", "content": final_answer, "sources": all_results}
    
    async def _synthesize_answer(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Synthesize final answer from all results"""
        prompt = f"""
        Question: {query}
        
        Retrieved information:
        {json.dumps(results, indent=2)}
        
        Synthesize a comprehensive answer to the question based on the retrieved information.
        """
        
        response = await self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        
        return response.choices[0].message.content
