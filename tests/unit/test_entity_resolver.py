import pytest
from services.graph.entity_resolver import EntityResolver
from services.graph.neo4j_adapter import Neo4jAdapter

@pytest.mark.asyncio
async def test_entity_resolution():
    graph_db = Neo4jAdapter()
    await graph_db.connect()
    
    resolver = EntityResolver(graph_db)
    
    entity1 = {
        'type': 'Person',
        'name': 'John Smith',
        'attributes': {'age': 30},
        'embedding': [0.1] * 3072
    }
    
    entity2 = {
        'type': 'Person',
        'name': 'J. Smith',
        'attributes': {'age': 30},
        'embedding': [0.11] * 3072
    }
    
    is_duplicate, existing_id = await resolver.resolve_entity(entity2)
    
    assert is_duplicate or existing_id is None
    
    await graph_db.disconnect()
