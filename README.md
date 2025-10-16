# Knowledge Graph Platform
Production-grade agentic retrieval system with automatic graph construction.


=============================

### Problem Statement

Build an extensible, production-grade platform that unifies knowledge from multiple sources into an intelligent retrieval system. The platform must: 

1. **Automatically construct knowledge graphs** from unstructured documents using LLM-generated ontologies and OpenAI embeddings, with a visual editor for ontology refinement and retrieval testing. 
2. **Provide a unified retrieval server** that combines three complementary search methods—vector similarity search using OpenAI embeddings for semantic matching, graph traversal for relationship-based queries, and logical filtering for metadata/attribute constraints—all orchestrated by autonomous AI agents that dynamically determine optimal retrieval strategies based on query complexity, enabling users to extract insights through natural language queries that seamlessly blend semantic understanding, relational reasoning, and precise filtering in a single, cohesive system.

### Requirements

### **1. Document-to-Graph Pipeline**

- LLM-powered automatic ontology generation (entities, relationships, hierarchies)
- OpenAI embedding integration for all graph elements
- Automated knowledge graph construction with entity resolution
- Visual ontology editor with LLM-assisted modifications
- Entity resolution
- Entity Deduplication

### **2. Agentic Retrieval System**

- Dynamic tool selection (vector search, graph traversal, logical filtering, Cypher/Gremlin generation)
- Multi-step reasoning with iterative refinement
- Streaming responses with reasoning chains
- a common extensible interface with integration of neo4j and aws netune

### Evaluation Criteria
 
| Category                               | Focus                                                                                                                | Weight  |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------- |
| **A. System Architecture**             | Modular services; Neo4j/Neptune parity; embedding store; entity resolution & dedup subsystems.                       | **25%** |
| **B. Graph Quality & Ontology**        | Ontology accuracy/completeness; entity resolution quality; relationship extraction; LLM-assisted refinement.         | **25%** |
| **C. Retrieval Intelligence**          | Agent routing across vector/graph/filter; hybrid relevance; latency; Cypher/Gremlin generation; streaming reasoning. | **25%** |
| **D. Extensibility & Maintainability** | Pluggable GraphDBs; clean APIs/SDKs; versioned ontology; CI/CD and test coverage; operability.                       | **25%** |
