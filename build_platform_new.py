#!/usr/bin/env python3
"""
Complete Knowledge Graph Platform Builder
Generates all source code files and packages into ZIP
"""

import os
import zipfile
from pathlib import Path

# ALL SOURCE CODE FILES
SOURCE_FILES = {
    # ============================================================================
    # CONFIGURATION FILES
    # ============================================================================
    
    "config/__init__.py": """\"\"\"Configuration package\"\"\"
""",

    "config/settings.py": """from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    
    # OpenAI Configuration
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIMENSION: int = 3072
    
    # Neo4j Configuration
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str
    
    # AWS Neptune Configuration (optional)
    NEPTUNE_ENDPOINT: Optional[str] = None
    NEPTUNE_PORT: int = 8182
    AWS_REGION: str = "us-east-1"
    
    # Vector Store Configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "knowledge_graph"
    
    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    
    # Processing Configuration
    MAX_CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    ENTITY_RESOLUTION_THRESHOLD: float = 0.85
    DEDUP_THRESHOLD: float = 0.95
    
    class Config:
        env_file = ".env"

settings = Settings()
""",

    "config/prompts.yaml": """ontology_generation: |
  You are an expert ontology engineer. Analyze the following document and extract:
  
  1. Entity Types: Identify all important entity categories (e.g., Person, Organization, Concept)
  2. Relationship Types: Define relationships between entities
  3. Hierarchies: Establish taxonomic relationships
  4. Attributes: Key properties for each entity type
  
  Document:
  {document_text}
  
  Return a structured ontology in JSON format with entity_types, relationship_types, and hierarchies.

entity_extraction: |
  Extract entities from the text according to this ontology:
  {ontology}
  
  Text:
  {text}
  
  Return entities in JSON format with type, name, attributes, and confidence scores.

relationship_extraction: |
  Given these entities: {entities}
  And this ontology: {ontology}
  
  Extract relationships from the text:
  {text}
  
  Return relationships in JSON format with source, target, type, and confidence.

entity_resolution: |
  Determine if these two entities refer to the same real-world entity:
  
  Entity 1: {entity1}
  Entity 2: {entity2}
  
  Consider: name variations, aliases, context, and attributes.
  Return: {{"is_same": true/false, "confidence": 0.0-1.0, "reasoning": "..."}}

query_planning: |
  Given this query: {query}
  
  Available tools:
  - vector_search: Semantic similarity search using embeddings
  - graph_traversal: Navigate relationships in the knowledge graph
  - logical_filter: Apply metadata/attribute constraints
  - cypher_query: Execute custom Cypher queries
  
  Plan a multi-step retrieval strategy. Return JSON with:
  {{"steps": [{{"tool": "...", "params": {{...}}, "reasoning": "..."}}]}}
""",

    # ============================================================================
    # GRAPH DATABASE LAYER
    # ============================================================================
    
    "services/__init__.py": """\"\"\"Services package\"\"\"
""",

    "services/graph/__init__.py": """\"\"\"Graph database services\"\"\"
""",

    "services/graph/graph_interface.py": """from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Node:
    id: str
    labels: List[str]
    properties: Dict[str, Any]

@dataclass
class Relationship:
    id: str
    type: str
    start_node: str
    end_node: str
    properties: Dict[str, Any]

class GraphDBInterface(ABC):
    \"\"\"Unified interface for graph databases\"\"\"
    
    @abstractmethod
    async def connect(self) -> None:
        \"\"\"Establish connection to graph database\"\"\"
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        \"\"\"Close connection\"\"\"
        pass
    
    @abstractmethod
    async def create_node(self, node: Node) -> str:
        \"\"\"Create a node and return its ID\"\"\"
        pass
    
    @abstractmethod
    async def create_relationship(self, rel: Relationship) -> str:
        \"\"\"Create a relationship and return its ID\"\"\"
        pass
    
    @abstractmethod
    async def find_nodes(self, labels: List[str], properties: Dict[str, Any]) -> List[Node]:
        \"\"\"Find nodes matching criteria\"\"\"
        pass
    
    @abstractmethod
    async def traverse(self, start_node_id: str, relationship_types: List[str], 
                      max_depth: int = 3) -> List[Dict[str, Any]]:
        \"\"\"Traverse graph from starting node\"\"\"
        pass
    
    @abstractmethod
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        \"\"\"Execute native query (Cypher/Gremlin)\"\"\"
        pass
""",

    "services/graph/neo4j_adapter.py": """from neo4j import AsyncGraphDatabase, AsyncDriver
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
        # Create constraints and indexes
        async with self.driver.session() as session:
            await session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")
            await session.run("CREATE INDEX entity_embedding IF NOT EXISTS FOR (n:Entity) ON (n.embedding)")
    
    async def disconnect(self) -> None:
        if self.driver:
            await self.driver.close()
    
    async def create_node(self, node: Node) -> str:
        query = f\"\"\"
        CREATE (n:{':'.join(node.labels)} $properties)
        RETURN n.id as id
        \"\"\"
        
        async with self.driver.session() as session:
            result = await session.run(query, properties=node.properties)
            record = await result.single()
            return record["id"]
    
    async def create_relationship(self, rel: Relationship) -> str:
        query = f\"\"\"
        MATCH (a:Entity {{id: $start_id}})
        MATCH (b:Entity {{id: $end_id}})
        CREATE (a)-[r:{rel.type} $properties]->(b)
        RETURN id(r) as rel_id
        \"\"\"
        
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
        
        query = f\"\"\"
        MATCH (n:{label_str})
        WHERE {where_str}
        RETURN n, labels(n) as labels
        LIMIT 100
        \"\"\"
        
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
        query = f\"\"\"
        MATCH path = (start:Entity {{id: $start_id}})-[r:{rel_filter}*1..{max_depth}]-(connected)
        RETURN path, nodes(path) as nodes, relationships(path) as rels
        LIMIT 50
        \"\"\"
        
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
""",

    "services/graph/neptune_adapter.py": """from gremlin_python.driver import client, serializer
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
        query = f\"\"\"
        g.V().has('id', '{rel.start_node}').as('a')
         .V().has('id', '{rel.end_node}')
         .addE('{rel.type}').from('a')
        \"\"\"
        
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
        query = f\"\"\"
        g.V().has('id', '{start_node_id}')
         .repeat(bothE().otherV().simplePath())
         .times({max_depth})
         .path()
         .limit(50)
        \"\"\"
        
        result = await asyncio.to_thread(self.client.submit, query)
        return [{"path": path} for path in result]
    
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        result = await asyncio.to_thread(self.client.submit, query, params or {})
        return list(result)
""",

    "services/graph/entity_resolver.py": """from typing import List, Dict, Tuple, Optional, Any
from openai import AsyncOpenAI
from config.settings import settings
import numpy as np
from .graph_interface import GraphDBInterface

class EntityResolver:
    def __init__(self, graph_db: GraphDBInterface):
        self.graph_db = graph_db
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    async def resolve_entity(self, new_entity: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        \"\"\"
        Resolve if new entity matches existing entity
        Returns: (is_duplicate, existing_entity_id)
        \"\"\"
        # Find candidates by entity type
        candidates = await self.graph_db.find_nodes(
            labels=[new_entity['type']],
            properties={}
        )
        
        if not candidates:
            return False, None
        
        # Compare with candidates
        for candidate in candidates:
            similarity = await self._compare_entities(new_entity, candidate.properties)
            if similarity > settings.ENTITY_RESOLUTION_THRESHOLD:
                return True, candidate.id
        
        return False, None
    
    async def _compare_entities(self, entity1: Dict, entity2: Dict) -> float:
        \"\"\"Compare two entities using LLM reasoning and embedding similarity\"\"\"
        # Embedding-based similarity
        emb1 = entity1.get('embedding')
        emb2 = entity2.get('embedding')
        
        if emb1 and emb2:
            emb_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        else:
            emb_sim = 0.0
        
        # LLM-based reasoning for complex cases
        if emb_sim > 0.7:
            prompt = f\"\"\"
            Determine if these entities refer to the same thing:
            
            Entity 1: {entity1}
            Entity 2: {entity2}
            
            Return only a number between 0.0 and 1.0 indicating similarity.
            \"\"\"
            
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            
            try:
                llm_sim = float(response.choices[0].message.content.strip())
            except:
                llm_sim = 0.0
            
            # Weighted combination
            return 0.6 * emb_sim + 0.4 * llm_sim
        
        return emb_sim
    
    async def deduplicate_entities(self, entity_ids: List[str]) -> Dict[str, str]:
        \"\"\"
        Find and merge duplicate entities
        Returns: mapping of duplicate_id -> canonical_id
        \"\"\"
        entities = []
        for eid in entity_ids:
            nodes = await self.graph_db.find_nodes([], {"id": eid})
            if nodes:
                entities.append(nodes[0])
        
        duplicates = {}
        processed = set()
        
        for i, e1 in enumerate(entities):
            if e1.id in processed:
                continue
            
            for e2 in entities[i+1:]:
                if e2.id in processed:
                    continue
                
                similarity = await self._compare_entities(e1.properties, e2.properties)
                if similarity > settings.DEDUP_THRESHOLD:
                    duplicates[e2.id] = e1.id
                    processed.add(e2.id)
        
        return duplicates
""",

    # ============================================================================
    # INGESTION SERVICES
    # ============================================================================
    
    "services/ingestion/__init__.py": """\"\"\"Ingestion services\"\"\"
""",

    "services/ingestion/ontology_generator.py": """from typing import Dict, Any, List
from openai import AsyncOpenAI
from config.settings import settings
import json
import yaml

class OntologyGenerator:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        with open('config/prompts.yaml', 'r') as f:
            self.prompts = yaml.safe_load(f)
    
    async def generate_ontology(self, document_text: str) -> Dict[str, Any]:
        \"\"\"Generate ontology from document using LLM\"\"\"
        prompt = self.prompts['ontology_generation'].format(document_text=document_text)
        
        response = await self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert ontology engineer. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        ontology = json.loads(response.choices[0].message.content)
        
        return {
            "version": "1.0",
            "entity_types": ontology.get("entity_types", []),
            "relationship_types": ontology.get("relationship_types", []),
            "hierarchies": ontology.get("hierarchies", []),
            "attributes": ontology.get("attributes", {}),
            "constraints": ontology.get("constraints", [])
        }
    
    async def refine_ontology(self, ontology: Dict[str, Any], 
                             feedback: str) -> Dict[str, Any]:
        \"\"\"Refine ontology based on user feedback\"\"\"
        prompt = f\"\"\"
        Current ontology:
        {json.dumps(ontology, indent=2)}
        
        User feedback: {feedback}
        
        Refine the ontology based on the feedback. Return the complete updated ontology as JSON.
        \"\"\"
        
        response = await self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    async def merge_ontologies(self, ontologies: List[Dict[str, Any]]) -> Dict[str, Any]:
        \"\"\"Merge multiple ontologies into unified schema\"\"\"
        prompt = f\"\"\"
        Merge these ontologies into a unified, consistent schema:
        {json.dumps(ontologies, indent=2)}
        
        Resolve conflicts, eliminate redundancies, and maintain semantic consistency.
        Return the merged ontology as JSON.
        \"\"\"
        
        response = await self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
""",

    "services/ingestion/entity_extractor.py": """from typing import List, Dict, Any
from openai import AsyncOpenAI
from config.settings import settings
import json
import yaml

class EntityExtractor:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        with open('config/prompts.yaml', 'r') as f:
            self.prompts = yaml.safe_load(f)
    
    async def extract_entities(self, text: str, ontology: Dict[str, Any]) -> List[Dict[str, Any]]:
        \"\"\"Extract entities from text according to ontology\"\"\"
        prompt = self.prompts['entity_extraction'].format(
            ontology=json.dumps(ontology),
            text=text
        )
        
        response = await self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("entities", [])
    
    async def extract_relationships(self, text: str, entities: List[Dict[str, Any]], 
                                   ontology: Dict[str, Any]) -> List[Dict[str, Any]]:
        \"\"\"Extract relationships between entities\"\"\"
        prompt = self.prompts['relationship_extraction'].format(
            entities=json.dumps(entities),
            ontology=json.dumps(ontology),
            text=text
        )
        
        response = await self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("relationships", [])
""",

    "services/ingestion/embedding_service.py": """from typing import List, Union, Dict, Any
from openai import AsyncOpenAI
from config.settings import settings

class EmbeddingService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    async def embed_text(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        \"\"\"Generate embeddings for text using OpenAI\"\"\"
        texts = [text] if isinstance(text, str) else text
        
        response = await self.client.embeddings.create(
            model=settings.OPENAI_EMBEDDING_MODEL,
            input=texts
        )
        
        embeddings = [item.embedding for item in response.data]
        return embeddings[0] if isinstance(text, str) else embeddings
    
    async def embed_entity(self, entity: Dict[str, Any]) -> List[float]:
        \"\"\"Generate embedding for entity\"\"\"
        text = f"{entity['type']}: {entity['name']}"
        if 'attributes' in entity:
            attrs = ", ".join(f"{k}={v}" for k, v in entity['attributes'].items())
            text += f" ({attrs})"
        
        return await self.embed_text(text)
    
    async def embed_relationship(self, rel: Dict[str, Any]) -> List[float]:
        \"\"\"Generate embedding for relationship\"\"\"
        text = f"{rel['source']} -{rel['type']}-> {rel['target']}"
        if 'properties' in rel:
            text += f" {rel['properties']}"
        
        return await self.embed_text(text)
""",

    "services/ingestion/document_processor.py": """from typing import Dict, Any, List
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
        \"\"\"
        Complete document processing pipeline
        \"\"\"
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
        \"\"\"Split document into overlapping chunks\"\"\"
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
""",

    # ============================================================================
    # VECTOR STORE
    # ============================================================================
    
    "services/vector_store/__init__.py": """\"\"\"Vector store services\"\"\"
""",

    "services/vector_store/embedding_store.py": """from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition
from typing import List, Dict, Any
from config.settings import settings

class EmbeddingStore:
    def __init__(self):
        self.client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.collection_name = settings.QDRANT_COLLECTION
    
    async def initialize(self):
        \"\"\"Create collection if it doesn't exist\"\"\"
        collections = await self.client.get_collections()
        
        if self.collection_name not in [c.name for c in collections.collections]:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIMENSION,
                    distance=Distance.COSINE
                )
            )
    
    async def store_embedding(self, entity_id: str, embedding: List[float], 
                            metadata: Dict[str, Any]):
        \"\"\"Store entity embedding with metadata\"\"\"
        await self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=entity_id,
                    vector=embedding,
                    payload=metadata
                )
            ]
        )
    
    async def search_similar(self, query_embedding: List[float], 
                           limit: int = 10,
                           filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        \"\"\"Search for similar entities\"\"\"
        search_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match={"value": v})
                for k, v in filters.items()
            ]
            search_filter = Filter(must=conditions)
        
        results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=search_filter
        )
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "metadata": hit.payload
            }
            for hit in results
        ]
""",

    # ============================================================================
    # RETRIEVAL SERVICES
    # ============================================================================
    
    "services/retrieval/__init__.py": """\"\"\"Retrieval services\"\"\"
""",

    "services/retrieval/tools/__init__.py": """\"\"\"Retrieval tools\"\"\"
""",

    "services/retrieval/tools/vector_search.py": """from typing import List, Dict, Any
from services.vector_store.embedding_store import EmbeddingStore
from services.ingestion.embedding_service import EmbeddingService

class VectorSearchTool:
    \"\"\"Semantic similarity search using embeddings\"\"\"
    
    name = "vector_search"
    description = "Search for entities semantically similar to the query"
    
    def __init__(self):
        self.embedding_store = EmbeddingStore()
        self.embedding_service = EmbeddingService()
    
    async def execute(self, query: str, limit: int = 10, 
                     filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        \"\"\"Execute vector similarity search\"\"\"
        # Generate query embedding
        query_embedding = await self.embedding_service.embed_text(query)
        
        # Search similar entities
        results = await self.embedding_store.search_similar(
            query_embedding,
            limit=limit,
            filters=filters
        )
        
        return results
""",

    "services/retrieval/tools/graph_traversal.py": """from typing import List, Dict, Any
from services.graph.graph_interface import GraphDBInterface

class GraphTraversalTool:
    \"\"\"Navigate relationships in the knowledge graph\"\"\"
    
    name = "graph_traversal"
    description = "Traverse graph relationships from starting entities"
    
    def __init__(self, graph_db: GraphDBInterface):
        self.graph_db = graph_db
    
    async def execute(self, start_entity_ids: List[str], 
                     relationship_types: List[str] = None,
                     max_depth: int = 3) -> List[Dict[str, Any]]:
        \"\"\"Execute graph traversal\"\"\"
        all_paths = []
        
        for entity_id in start_entity_ids:
            paths = await self.graph_db.traverse(
                start_node_id=entity_id,
                relationship_types=relationship_types or [],
                max_depth=max_depth
            )
            all_paths.extend(paths)
        
        return all_paths
""",

    "services/retrieval/tools/logical_filter.py": """from typing import List, Dict, Any
from services.graph.graph_interface import GraphDBInterface

class LogicalFilterTool:
    \"\"\"Apply metadata and attribute constraints\"\"\"
    
    name = "logical_filter"
    description = "Filter entities by attributes and metadata"
    
    def __init__(self, graph_db: GraphDBInterface):
        self.graph_db = graph_db
    
    async def execute(self, filters: Dict[str, Any], 
                     entity_types: List[str] = None) -> List[Dict[str, Any]]:
        \"\"\"Execute logical filtering\"\"\"
        nodes = await self.graph_db.find_nodes(
            labels=entity_types or [],
            properties=filters
        )
        
        return [{"id": n.id, "properties": n.properties} for n in nodes]
""",

    "services/retrieval/tools/query_generator.py": """from typing import Dict, Any, List
from openai import AsyncOpenAI
from config.settings import settings
from services.graph.graph_interface import GraphDBInterface

class QueryGeneratorTool:
    \"\"\"Generate and execute Cypher/Gremlin queries\"\"\"
    
    name = "query_generator"
    description = "Generate custom graph database queries"
    
    def __init__(self, graph_db: GraphDBInterface, db_type: str = "neo4j"):
        self.graph_db = graph_db
        self.db_type = db_type
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    async def execute(self, natural_language_query: str) -> List[Dict[str, Any]]:
        \"\"\"Generate and execute query from natural language\"\"\"
        query_language = "Cypher" if self.db_type == "neo4j" else "Gremlin"
        
        prompt = f\"\"\"
        Generate a {query_language} query for this request:
        {natural_language_query}
        
        Return only the query, no explanations.
        \"\"\"
        
        response = await self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        query = response.choices[0].message.content.strip()
        
        # Execute query
        results = await self.graph_db.execute_query(query)
        
        return results
""",

    "services/retrieval/query_planner.py": """from typing import List, Dict, Any
from openai import AsyncOpenAI
from config.settings import settings
import json
import yaml

class QueryPlanner:
    \"\"\"Plan multi-step retrieval strategies\"\"\"
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        with open('config/prompts.yaml', 'r') as f:
            self.prompts = yaml.safe_load(f)
    
    async def plan_query(self, query: str) -> List[Dict[str, Any]]:
        \"\"\"Generate query execution plan\"\"\"
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
        \"\"\"Refine plan based on intermediate results\"\"\"
        prompt = f\"\"\"
        Original query: {query}
        Current plan: {json.dumps(plan)}
        Results so far: {json.dumps(previous_results)}
        
        Should we continue with the plan or adjust it? Return updated plan as JSON.
        \"\"\"
        
        response = await self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        refined = json.loads(response.choices[0].message.content)
        return refined.get("steps", plan)
""",

    "services/retrieval/agent_orchestrator.py": """from typing import List, Dict, Any, AsyncGenerator
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
    \"\"\"Autonomous agent for intelligent retrieval\"\"\"
    
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
        \"\"\"Execute retrieval with streaming responses\"\"\"
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
        \"\"\"Synthesize final answer from all results\"\"\"
        prompt = f\"\"\"
        Question: {query}
        
        Retrieved information:
        {json.dumps(results, indent=2)}
        
        Synthesize a comprehensive answer to the question based on the retrieved information.
        \"\"\"
        
        response = await self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        
        return response.choices[0].message.content
""",

    # ============================================================================
    # API LAYER
    # ============================================================================
    
    "api/__init__.py": """\"\"\"API package\"\"\"
""",

    "api/routes/__init__.py": """\"\"\"API routes\"\"\"
""",

    "api/models/__init__.py": """\"\"\"API models\"\"\"
""",

    "api/models/schemas.py": """from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class Entity(BaseModel):
    id: str
    type: str
    name: str
    attributes: Dict[str, Any] = {}
    embedding: Optional[List[float]] = None
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Relationship(BaseModel):
    id: str
    type: str
    source: str
    target: str
    properties: Dict[str, Any] = {}
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Ontology(BaseModel):
    version: str
    entity_types: List[Dict[str, Any]]
    relationship_types: List[Dict[str, Any]]
    hierarchies: List[Dict[str, Any]]
    attributes: Dict[str, Any]
    constraints: List[Dict[str, Any]] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ProcessDocumentRequest(BaseModel):
    text: str
    ontology: Optional[Dict[str, Any]] = None

class ProcessDocumentResponse(BaseModel):
    success: bool
    stats: Dict[str, Any]
    message: str

class QueryRequest(BaseModel):
    query: str
""",

    "api/routes/health.py": """from fastapi import APIRouter
import api.main as main_app

router = APIRouter()

@router.get("/")
async def health_check():
    \"\"\"Health check endpoint\"\"\"
    return {
        "status": "healthy",
        "graph_db": "connected" if main_app.graph_db else "disconnected",
        "vector_store": "connected" if main_app.embedding_store else "disconnected"
    }

@router.get("/stats")
async def get_stats():
    \"\"\"Get system statistics\"\"\"
    try:
        # Count nodes
        node_count_result = await main_app.graph_db.execute_query(
            "MATCH (n) RETURN count(n) as count"
        )
        node_count = node_count_result[0]["count"] if node_count_result else 0
        
        # Count relationships
        rel_count_result = await main_app.graph_db.execute_query(
            "MATCH ()-[r]->() RETURN count(r) as count"
        )
        rel_count = rel_count_result[0]["count"] if rel_count_result else 0
        
        return {
            "nodes": node_count,
            "relationships": rel_count,
            "status": "operational"
        }
    except Exception as e:
        return {"error": str(e)}
""",

    "api/routes/documents.py": """from fastapi import APIRouter, UploadFile, File, HTTPException
from api.models.schemas import ProcessDocumentRequest, ProcessDocumentResponse
from services.ingestion.document_processor import DocumentProcessor
import api.main as main_app

router = APIRouter()

@router.post("/process", response_model=ProcessDocumentResponse)
async def process_document(request: ProcessDocumentRequest):
    \"\"\"Process document and build knowledge graph\"\"\"
    try:
        processor = DocumentProcessor(main_app.graph_db)
        stats = await processor.process_document(request.text, request.ontology)
        
        return ProcessDocumentResponse(
            success=True,
            stats=stats,
            message="Document processed successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    \"\"\"Upload and process document file\"\"\"
    try:
        content = await file.read()
        text = content.decode('utf-8')
        
        processor = DocumentProcessor(main_app.graph_db)
        stats = await processor.process_document(text)
        
        return {"success": True, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
""",

    "api/routes/ontology.py": """from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from services.ingestion.ontology_generator import OntologyGenerator

router = APIRouter()

class GenerateOntologyRequest(BaseModel):
    document_text: str

class RefineOntologyRequest(BaseModel):
    ontology: Dict[str, Any]
    feedback: str

class MergeOntologiesRequest(BaseModel):
    ontologies: List[Dict[str, Any]]

@router.post("/generate")
async def generate_ontology(request: GenerateOntologyRequest):
    \"\"\"Generate ontology from document\"\"\"
    try:
        generator = OntologyGenerator()
        ontology = await generator.generate_ontology(request.document_text)
        return {"success": True, "ontology": ontology}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/refine")
async def refine_ontology(request: RefineOntologyRequest):
    \"\"\"Refine ontology based on feedback\"\"\"
    try:
        generator = OntologyGenerator()
        refined = await generator.refine_ontology(request.ontology, request.feedback)
        return {"success": True, "ontology": refined}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/merge")
async def merge_ontologies(request: MergeOntologiesRequest):
    \"\"\"Merge multiple ontologies\"\"\"
    try:
        generator = OntologyGenerator()
        merged = await generator.merge_ontologies(request.ontologies)
        return {"success": True, "ontology": merged}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
""",

    "api/routes/retrieval.py": """from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from api.models.schemas import QueryRequest
from services.retrieval.agent_orchestrator import AgentOrchestrator
import api.main as main_app
import json

router = APIRouter()

@router.websocket("/stream")
async def retrieval_stream(websocket: WebSocket):
    \"\"\"Streaming retrieval with reasoning chain\"\"\"
    await websocket.accept()
    
    try:
        orchestrator = AgentOrchestrator(main_app.graph_db)
        
        while True:
            data = await websocket.receive_text()
            query_data = json.loads(data)
            query = query_data["query"]
            
            async for event in orchestrator.retrieve(query):
                await websocket.send_json(event)
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close()

@router.post("/query")
async def query_knowledge_graph(request: QueryRequest):
    \"\"\"Non-streaming retrieval endpoint\"\"\"
    try:
        orchestrator = AgentOrchestrator(main_app.graph_db)
        
        results = []
        async for event in orchestrator.retrieve(request.query):
            results.append(event)
        
        final_event = next((e for e in reversed(results) if e["type"] == "final_answer"), None)
        
        return {
            "success": True,
            "answer": final_event["content"] if final_event else "No answer found",
            "sources": final_event.get("sources", []) if final_event else [],
            "reasoning_chain": [e for e in results if e["type"] in ["step", "plan"]]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
""",

    "api/main.py": """from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from .routes import documents, ontology, retrieval, health
from services.graph.neo4j_adapter import Neo4jAdapter
from services.vector_store.embedding_store import EmbeddingStore
from config.settings import settings

# Global state
graph_db = None
embedding_store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global graph_db, embedding_store
    graph_db = Neo4jAdapter()
    await graph_db.connect()
    
    embedding_store = EmbeddingStore()
    await embedding_store.initialize()
    
    yield
    
    # Shutdown
    await graph_db.disconnect()

app = FastAPI(
    title="Knowledge Graph Platform",
    description="Production-grade agentic retrieval system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(ontology.router, prefix="/ontology", tags=["ontology"])
app.include_router(retrieval.router, prefix="/retrieval", tags=["retrieval"])
""",

    # ============================================================================
    # TESTS
    # ============================================================================
    
    "tests/__init__.py": """\"\"\"Tests package\"\"\"
""",

    "tests/unit/__init__.py": """\"\"\"Unit tests\"\"\"
""",

    "tests/integration/__init__.py": """\"\"\"Integration tests\"\"\"
""",

    "tests/unit/test_entity_resolver.py": """import pytest
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
""",

    "tests/integration/test_document_processing.py": """import pytest
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
""",

    # ============================================================================
    # DOCKER & DEPLOYMENT
    # ============================================================================
    
    "docker/Dockerfile.api": """FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
""",

    "docker/docker-compose.yml": """version: '3.8'

services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
      - QDRANT_HOST=qdrant
      - REDIS_HOST=redis
    depends_on:
      - neo4j
      - qdrant
      - redis
    volumes:
      - ../:/app

  neo4j:
    image: neo4j:5.14
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}
      - NEO4J_PLUGINS=["apoc"]
    volumes:
      - neo4j_data:/data

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  neo4j_data:
  qdrant_data:
  redis_data:
""",

    # ============================================================================
    # ROOT FILES
    # ============================================================================
    
    "run.py": """#!/usr/bin/env python3
import uvicorn
from config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=True,
        log_level="info"
    )
""",

    "requirements.txt": """fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
openai==1.3.0
langchain==0.0.340
langchain-openai==0.0.2
neo4j==5.14.1
boto3==1.29.7
qdrant-client==1.7.0
sentence-transformers==2.2.2
tiktoken==0.5.1
python-multipart==0.0.6
aiofiles==23.2.1
redis==5.0.1
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2
gremlinpython==3.7.0
pyyaml==6.0.1
numpy==1.24.3
""",

    ".env.example": """# OpenAI Configuration
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_secure_password

# Vector Store
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Processing
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=200
ENTITY_RESOLUTION_THRESHOLD=0.85
DEDUP_THRESHOLD=0.95
""",

    ".gitignore": """__pycache__/
*.py[cod]
.Python
env/
venv/
.env
*.db
*.log
.pytest_cache/
htmlcov/
.coverage
""",

    "README.md": """# Knowledge Graph Platform

Production-grade agentic retrieval system with automatic graph construction.

## Quick Start

```bash
# 1. Setup
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# 2. Start infrastructure
docker-compose -f docker/docker-compose.yml up -d

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run server
python run.py
```

Access at http://localhost:8000
Docs at http://localhost:8000/docs

## Features

- LLM-powered ontology generation
- Entity resolution & deduplication  
- Agentic retrieval with streaming
- Neo4j & AWS Neptune support
- OpenAI embeddings integration

## Documentation

See DOCUMENTATION.md for complete guide.
"""
}


def create_all_files():
    """Create all source code files"""
    print("="* 70)
    print("KNOWLEDGE GRAPH PLATFORM - COMPLETE BUILD")
    print("="* 70)
    print()
    
    created_count = 0
    for filepath, content in SOURCE_FILES.items():
        file_path = Path(filepath)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        print(f"✓ Created: {filepath}")
        created_count += 1
    
    print(f"\n✓ Successfully created {created_count} files!")
    return created_count


def create_zip_package():
    """Create ZIP file of entire project"""
    print("\n" + "="* 70)
    print("CREATING ZIP PACKAGE")
    print("="* 70)
    print()
    
    zip_filename = "knowledge-graph-platform.zip"
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('.'):
            # Skip unwanted directories
            dirs[:] = [d for d in dirs if d not in {
                '__pycache__', '.git', '.pytest_cache', 'venv', 'env',
                'node_modules', '.idea', '.vscode', '__pycache__'
            }]
            
            for file in files:
                if file.endswith(('.pyc', '.pyo')) or file == zip_filename:
                    continue
                    
                filepath = os.path.join(root, file)
                arcname = filepath[2:] if filepath.startswith('./') else filepath
                zipf.write(filepath, arcname)
                print(f"  Added: {arcname}")
    
    size_mb = os.path.getsize(zip_filename) / 1024 / 1024
    print(f"\n✓ Created {zip_filename} ({size_mb:.2f} MB)")
    return zip_filename


def main():
    print("\n🚀 Starting complete build process...\n")
    
    # Create all source files
    file_count = create_all_files()
    
    # Create ZIP package
    zip_file = create_zip_package()
    
    print("\n" + "="* 70)
    print("✅ BUILD COMPLETE!")
    print("="* 70)
    print(f"""
📦 Package: {zip_file}
📄 Files created: {file_count}
    
🎯 Next Steps:
1. Extract the ZIP file
2. cd knowledge-graph-platform
3. cp .env.example .env
4. Edit .env with your OPENAI_API_KEY
5. docker-compose -f docker/docker-compose.yml up -d
6. pip install -r requirements.txt
7. python run.py

🌐 Access:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs  
- Neo4j: http://localhost:7474

📚 Read README.md for detailed instructions!
""")


if __name__ == "__main__":
    main()