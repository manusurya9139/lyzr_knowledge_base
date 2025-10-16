from gremlin_python.driver import client, serializer
from typing import List, Dict, Any
from .graph_interface import GraphDBInterface, Node, Relationship
from config.settings import settings
import asyncio

class NeptuneAdapter(GraphDBInterface):
    def __init__(self):
        self.client = None
        
    async def connect(self) -> None:
        endpoint = f"wss://{settings.NEPTUNE_ENDPOINT}:{settings.NEPTUNE_PORT}/gremlin"
        self.client = client.Client(
            endpoint,
            'g',
            message_serializer=serializer.GraphSONSerializersV2d0()
        )
    
    async def disconnect(self) -> None:
        if self.client:
            self.client.close()
    
    async def create_node(self, node: Node) -> str:
        props = []
        for k, v in node.properties.items():
            props.extend([k, v])
        
        query = f"g.addV('{node.labels[0]}').property('id', '{node.properties['id']}')"
        for i in range(0, len(props), 2):
            if props[i] != 'id':
                query += f".property('{props[i]}', '{props[i+1]}')"
        
        result = await asyncio.to_thread(self.client.submit, query)
        return node.properties['id']
    
    async def create_relationship(self, rel: Relationship) -> str:
        query = f"""
        g.V().has('id', '{rel.start_node}').as('a')
         .V().has('id', '{rel.end_node}')
         .addE('{rel.type}').from('a')
        """
        
        result = await asyncio.to_thread(self.client.submit, query)
        return rel.id
    
    async def find_nodes(self, labels: List[str], properties: Dict[str, Any]) -> List[Node]:
        query = f"g.V().hasLabel('{labels[0] if labels else 'Entity'}')"
        for k, v in properties.items():
            query += f".has('{k}', '{v}')"
        query += ".limit(100).valueMap(true)"
        
        result = await asyncio.to_thread(self.client.submit, query)
        nodes = []
        for data in result:
            nodes.append(Node(
                id=data['id'],
                labels=labels,
                properties={k: v[0] if isinstance(v, list) else v for k, v in data.items()}
            ))
        return nodes
    
    async def traverse(self, start_node_id: str, relationship_types: List[str], 
                      max_depth: int = 3) -> List[Dict[str, Any]]:
        query = f"""
        g.V().has('id', '{start_node_id}')
         .repeat(bothE().otherV().simplePath())
         .times({max_depth})
         .path()
         .limit(50)
        """
        
        result = await asyncio.to_thread(self.client.submit, query)
        return [{"path": path} for path in result]
    
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        result = await asyncio.to_thread(self.client.submit, query, params or {})
        return list(result)
