from typing import List, Dict, Any
from services.graph.graph_interface import GraphDBInterface

class GraphTraversalTool:
    """Navigate relationships in the knowledge graph"""
    
    name = "graph_traversal"
    description = "Traverse graph relationships from starting entities"
    
    def __init__(self, graph_db: GraphDBInterface):
        self.graph_db = graph_db
    
    async def execute(self, start_entity_ids: List[str], 
                     relationship_types: List[str] = None,
                     max_depth: int = 3) -> List[Dict[str, Any]]:
        """Execute graph traversal"""
        all_paths = []
        
        for entity_id in start_entity_ids:
            paths = await self.graph_db.traverse(
                start_node_id=entity_id,
                relationship_types=relationship_types or [],
                max_depth=max_depth
            )
            all_paths.extend(paths)
        
        return all_paths
