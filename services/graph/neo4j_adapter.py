from neo4j import AsyncGraphDatabase, AsyncDriver
from typing import List, Dict, Any
from .graph_interface import GraphDBInterface, Node, Relationship
from config.settings import settings

class Neo4jAdapter(GraphDBInterface):
    def __init__(self):
        self.driver: AsyncDriver = None
        
    async def connect(self) -> None:
        self.driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )
        # Create constraints and indexes with retry logic
        await self._create_constraints_and_indexes()
    
    async def _create_constraints_and_indexes(self) -> None:
        """Create constraints and indexes with retry logic for deadlocks"""
        import asyncio
        from neo4j.exceptions import TransientError
        
        max_retries = 5
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                async with self.driver.session() as session:
                    # Use separate transactions for each constraint/index
                    await session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")
                    await session.run("CREATE INDEX entity_embedding IF NOT EXISTS FOR (n:Entity) ON (n.embedding)")
                break  # Success, exit retry loop
            except TransientError as e:
                if "DeadlockDetected" in str(e) and attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    raise  # Re-raise if not a deadlock or max retries reached
            except Exception as e:
                # For other errors, log and continue (constraints might already exist)
                print(f"Warning: Could not create constraint/index: {e}")
                break
    
    async def disconnect(self) -> None:
        if self.driver:
            await self.driver.close()
    
    async def create_node(self, node: Node) -> str:
        query = f"""
        CREATE (n:{':'.join(node.labels)} $properties)
        RETURN n.id as id
        """
        
        async with self.driver.session() as session:
            result = await session.run(query, properties=node.properties)
            record = await result.single()
            return record["id"]
    
    async def create_relationship(self, rel: Relationship) -> str:
        query = f"""
        MATCH (a:Entity {{id: $start_id}})
        MATCH (b:Entity {{id: $end_id}})
        CREATE (a)-[r:{rel.type} $properties]->(b)
        RETURN id(r) as rel_id
        """
        
        async with self.driver.session() as session:
            result = await session.run(
                query,
                start_id=rel.start_node,
                end_id=rel.end_node,
                properties=rel.properties
            )
            record = await result.single()
            return str(record["rel_id"])
    
    async def find_nodes(self, labels: List[str], properties: Dict[str, Any]) -> List[Node]:
        label_str = ":".join(labels) if labels else "Entity"
        where_clauses = [f"n.{k} = ${k}" for k in properties.keys()]
        where_str = " AND ".join(where_clauses) if where_clauses else "true"
        
        query = f"""
        MATCH (n:{label_str})
        WHERE {where_str}
        RETURN n, labels(n) as labels
        LIMIT 100
        """
        
        async with self.driver.session() as session:
            result = await session.run(query, **properties)
            nodes = []
            async for record in result:
                nodes.append(Node(
                    id=record["n"]["id"],
                    labels=record["labels"],
                    properties=dict(record["n"])
                ))
            return nodes
    
    async def traverse(self, start_node_id: str, relationship_types: List[str], 
                      max_depth: int = 3) -> List[Dict[str, Any]]:
        rel_filter = "|".join(relationship_types) if relationship_types else ""
        query = f"""
        MATCH path = (start:Entity {{id: $start_id}})-[r:{rel_filter}*1..{max_depth}]-(connected)
        RETURN path, nodes(path) as nodes, relationships(path) as rels
        LIMIT 50
        """
        
        async with self.driver.session() as session:
            result = await session.run(query, start_id=start_node_id)
            paths = []
            async for record in result:
                paths.append({
                    "nodes": [dict(n) for n in record["nodes"]],
                    "relationships": [dict(r) for r in record["rels"]]
                })
            return paths
    
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        async with self.driver.session() as session:
            result = await session.run(query, **(params or {}))
            return [dict(record) async for record in result]
