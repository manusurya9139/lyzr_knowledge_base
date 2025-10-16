from typing import List, Dict, Any
from services.graph.graph_interface import GraphDBInterface

class LogicalFilterTool:
    """Apply metadata and attribute constraints"""
    
    name = "logical_filter"
    description = "Filter entities by attributes and metadata"
    
    def __init__(self, graph_db: GraphDBInterface):
        self.graph_db = graph_db
    
    async def execute(self, filters: Dict[str, Any], 
                     entity_types: List[str] = None) -> List[Dict[str, Any]]:
        """Execute logical filtering"""
        nodes = await self.graph_db.find_nodes(
            labels=entity_types or [],
            properties=filters
        )
        
        return [{"id": n.id, "properties": n.properties} for n in nodes]
