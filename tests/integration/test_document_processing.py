import pytest
from services.ingestion.document_processor import DocumentProcessor
from services.graph.neo4j_adapter import Neo4jAdapter

@pytest.mark.asyncio
async def test_full_document_pipeline():
    graph_db = Neo4jAdapter()
    await graph_db.connect()
    
    processor = DocumentProcessor(graph_db)
    
    document = '''
    Albert Einstein was a theoretical physicist who developed the theory of relativity.
    He worked at Princeton University and won the Nobel Prize in Physics in 1921.
    '''
    
    result = await processor.process_document(document)
    
    assert result['entities_created'] > 0
    assert result['relationships_created'] > 0
    assert 'ontology' in result
    
    await graph_db.disconnect()
