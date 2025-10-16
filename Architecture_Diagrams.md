# Knowledge Graph Platform - Architecture Diagrams

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Knowledge Graph Platform                     │
├─────────────────────────────────────────────────────────────────┤
│  API Layer (FastAPI)                                            │
│  ├── Health Monitoring                                          │
│  ├── Document Processing                                        │
│  ├── Ontology Management                                        │
│  └── Intelligent Retrieval                                     │
├─────────────────────────────────────────────────────────────────┤
│  Service Layer                                                  │
│  ├── Document Processor                                         │
│  ├── Ontology Generator                                         │
│  ├── Entity Extractor                                           │
│  ├── Embedding Service                                          │
│  ├── Entity Resolver                                            │
│  └── Agent Orchestrator                                         │
├─────────────────────────────────────────────────────────────────┤
│  Data Layer                                                     │
│  ├── Neo4j Graph Database                                       │
│  ├── Qdrant Vector Store                                        │
│  └── Redis Cache                                                │
├─────────────────────────────────────────────────────────────────┤
│  External Services                                              │
│  └── OpenAI API (GPT-4o, Embeddings)                          │
└─────────────────────────────────────────────────────────────────┘
```

## Component Interaction Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   Ontology     │    │   Entity        │
│   Processor     │───▶│   Generator    │───▶│   Extractor    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Embedding     │    │   Entity        │    │   Knowledge     │
│   Service       │    │   Resolver      │    │   Graph         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Vector Store  │    │   Graph DB      │    │   Agent         │
│   (Qdrant)      │    │   (Neo4j)       │    │   Orchestrator  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Data Processing Pipeline

```
Document Input
     │
     ▼
┌─────────────┐
│ Ontology    │ ◄─── LLM (GPT-4o)
│ Generation  │
└─────────────┘
     │
     ▼
┌─────────────┐
│ Entity      │ ◄─── LLM (GPT-4o)
│ Extraction  │
└─────────────┘
     │
     ▼
┌─────────────┐
│ Embedding   │ ◄─── OpenAI Embeddings
│ Generation  │
└─────────────┘
     │
     ▼
┌─────────────┐
│ Entity      │
│ Resolution  │
└─────────────┘
     │
     ▼
┌─────────────┐
│ Knowledge   │
│ Graph       │
└─────────────┘
```

## Retrieval System Architecture

```
User Query
     │
     ▼
┌─────────────┐
│ Query       │
│ Planner     │ ◄─── LLM (GPT-4o)
└─────────────┘
     │
     ▼
┌─────────────┐
│ Agent       │
│ Orchestrator│
└─────────────┘
     │
     ├─── Vector Search Tool ────▶ Qdrant
     ├─── Graph Traversal Tool ───▶ Neo4j
     ├─── Logical Filter Tool ───▶ Neo4j
     └─── Query Generator Tool ───▶ Neo4j
     │
     ▼
┌─────────────┐
│ Result      │
│ Synthesis   │ ◄─── LLM (GPT-4o)
└─────────────┘
     │
     ▼
Streaming Response
```

## Docker Services Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Compose Services                     │
├─────────────────────────────────────────────────────────────────┤
│  API Service (Port 8000)                                        │
│  ├── FastAPI Application                                        │
│  ├── Uvicorn Server                                            │
│  └── Health Checks                                             │
├─────────────────────────────────────────────────────────────────┤
│  Neo4j Service (Ports 7474, 7687)                              │
│  ├── Graph Database                                            │
│  ├── APOC Plugins                                              │
│  └── Health Checks                                             │
├─────────────────────────────────────────────────────────────────┤
│  Qdrant Service (Port 6333)                                     │
│  ├── Vector Database                                           │
│  └── Embedding Storage                                         │
├─────────────────────────────────────────────────────────────────┤
│  Redis Service (Port 6379)                                     │
│  ├── Cache Layer                                               │
│  └── Session Storage                                           │
└─────────────────────────────────────────────────────────────────┘
```

## API Endpoints Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Endpoints                           │
├─────────────────────────────────────────────────────────────────┤
│  Health Endpoints                                               │
│  ├── GET /health/                                               │
│  └── GET /health/stats                                          │
├─────────────────────────────────────────────────────────────────┤
│  Document Processing                                            │
│  ├── POST /documents/process                                    │
│  └── POST /documents/upload                                    │
├─────────────────────────────────────────────────────────────────┤
│  Ontology Management                                            │
│  ├── POST /ontology/generate                                    │
│  ├── POST /ontology/refine                                      │
│  └── POST /ontology/merge                                       │
├─────────────────────────────────────────────────────────────────┤
│  Intelligent Retrieval                                          │
│  ├── POST /retrieval/query                                      │
│  └── WS /retrieval/stream                                       │
└─────────────────────────────────────────────────────────────────┘
```

## Testing Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Testing Framework                       │
├─────────────────────────────────────────────────────────────────┤
│  Unit Tests                                                     │
│  ├── Entity Resolver Tests                                     │
│  ├── Graph Adapter Tests                                       │
│  └── Service Layer Tests                                       │
├─────────────────────────────────────────────────────────────────┤
│  Integration Tests                                              │
│  ├── Document Processing Pipeline                               │
│  ├── End-to-End Retrieval                                      │
│  └── Database Integration                                      │
├─────────────────────────────────────────────────────────────────┤
│  Manual Testing                                                 │
│  ├── API Endpoint Testing                                      │
│  ├── WebSocket Testing                                         │
│  └── Performance Testing                                       │
└─────────────────────────────────────────────────────────────────┘
```

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Production Deployment                       │
├─────────────────────────────────────────────────────────────────┤
│  Load Balancer (Nginx/HAProxy)                                 │
│  ├── SSL Termination                                           │
│  ├── Request Routing                                           │
│  └── Health Checks                                             │
├─────────────────────────────────────────────────────────────────┤
│  API Cluster                                                    │
│  ├── API Instance 1 (Port 8000)                               │
│  ├── API Instance 2 (Port 8001)                               │
│  └── API Instance N (Port 800N)                               │
├─────────────────────────────────────────────────────────────────┤
│  Database Cluster                                               │
│  ├── Neo4j Primary                                             │
│  ├── Neo4j Secondary                                           │
│  └── Neo4j Arbiter                                             │
├─────────────────────────────────────────────────────────────────┤
│  Vector Store Cluster                                           │
│  ├── Qdrant Node 1                                             │
│  ├── Qdrant Node 2                                             │
│  └── Qdrant Node N                                             │
├─────────────────────────────────────────────────────────────────┤
│  Cache Cluster                                                  │
│  ├── Redis Master                                              │
│  └── Redis Replicas                                            │
└─────────────────────────────────────────────────────────────────┘
```

## Security Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Security Layers                         │
├─────────────────────────────────────────────────────────────────┤
│  Network Security                                               │
│  ├── Firewall Rules                                             │
│  ├── VPN Access                                                 │
│  └── Network Segmentation                                       │
├─────────────────────────────────────────────────────────────────┤
│  Application Security                                           │
│  ├── API Authentication                                         │
│  ├── Input Validation                                           │
│  └── Rate Limiting                                              │
├─────────────────────────────────────────────────────────────────┤
│  Data Security                                                  │
│  ├── Encryption at Rest                                         │
│  ├── Encryption in Transit                                      │
│  └── Access Controls                                            │
├─────────────────────────────────────────────────────────────────┤
│  External Security                                              │
│  ├── OpenAI API Key Management                                  │
│  ├── Database Credentials                                       │
│  └── Secret Management                                          │
└─────────────────────────────────────────────────────────────────┘
```

## Monitoring and Observability

```
┌─────────────────────────────────────────────────────────────────┐
│                    Monitoring Stack                            │
├─────────────────────────────────────────────────────────────────┤
│  Application Metrics                                            │
│  ├── API Response Times                                         │
│  ├── Request Rates                                              │
│  ├── Error Rates                                                │
│  └── Resource Usage                                             │
├─────────────────────────────────────────────────────────────────┤
│  Infrastructure Metrics                                         │
│  ├── CPU Usage                                                  │
│  ├── Memory Usage                                               │
│  ├── Disk Usage                                                 │
│  └── Network Traffic                                            │
├─────────────────────────────────────────────────────────────────┤
│  Business Metrics                                               │
│  ├── Documents Processed                                        │
│  ├── Queries Executed                                           │
│  ├── Graph Size                                                 │
│  └── User Activity                                              │
├─────────────────────────────────────────────────────────────────┤
│  Alerting                                                       │
│  ├── Health Check Failures                                      │
│  ├── Performance Degradation                                    │
│  ├── Resource Exhaustion                                        │
│  └── Security Incidents                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

*These diagrams can be converted to visual representations using tools like:*
- *Mermaid Live Editor*
- *Draw.io*
- *Lucidchart*
- *Visio*
- *PlantUML*
