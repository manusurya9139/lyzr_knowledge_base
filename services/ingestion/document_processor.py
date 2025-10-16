from typing import Dict, Any, List
import uuid
from .ontology_generator import OntologyGenerator
from .entity_extractor import EntityExtractor
from .embedding_service import EmbeddingService
from services.graph.graph_interface import GraphDBInterface, Node, Relationship
from services.graph.entity_resolver import EntityResolver
from config.settings import settings

class DocumentProcessor:
    def __init__(self, graph_db: GraphDBInterface):
        self.graph_db = graph_db
        self.ontology_gen = OntologyGenerator()
        self.entity_extractor = EntityExtractor()
        self.embedding_service = EmbeddingService()
        self.entity_resolver = EntityResolver(graph_db)
    
    async def process_document(self, document_text: str, 
                              ontology: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Complete document processing pipeline
        """
        # Step 1: Ontology
        if ontology is None:
            ontology = await self.ontology_gen.generate_ontology(document_text)
        
        # Step 2: Chunk document
        chunks = self._chunk_document(document_text, settings.MAX_CHUNK_SIZE, 
                                     settings.CHUNK_OVERLAP)
        
        all_entities = []
        all_relationships = []
        
        # Step 3: Extract from each chunk
        for chunk in chunks:
            entities = await self.entity_extractor.extract_entities(chunk, ontology)
            
            # Generate embeddings for entities
            for entity in entities:
                entity['embedding'] = await self.embedding_service.embed_entity(entity)
                entity['id'] = str(uuid.uuid4())
            
            relationships = await self.entity_extractor.extract_relationships(
                chunk, entities, ontology
            )
            
            all_entities.extend(entities)
            all_relationships.extend(relationships)
        
        # Step 4: Entity resolution
        entity_map = {}
        
        for entity in all_entities:
            is_duplicate, existing_id = await self.entity_resolver.resolve_entity(entity)
            
            if is_duplicate:
                entity_map[entity['id']] = existing_id
            else:
                node = Node(
                    id=entity['id'],
                    labels=['Entity', entity['type']],
                    properties=entity
                )
                node_id = await self.graph_db.create_node(node)
                entity_map[entity['id']] = node_id
        
        # Step 5: Create relationships
        relationship_ids = []
        for rel in all_relationships:
            start_id = entity_map.get(rel['source'], rel['source'])
            end_id = entity_map.get(rel['target'], rel['target'])
            
            relationship = Relationship(
                id=str(uuid.uuid4()),
                type=rel['type'],
                start_node=start_id,
                end_node=end_id,
                properties=rel.get('properties', {})
            )
            
            rel_id = await self.graph_db.create_relationship(relationship)
            relationship_ids.append(rel_id)
        
        return {
            "ontology": ontology,
            "entities_created": len(set(entity_map.values())),
            "relationships_created": len(relationship_ids),
            "entities_resolved": len([v for k, v in entity_map.items() if k != v])
        }
    
    def _chunk_document(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split document into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
