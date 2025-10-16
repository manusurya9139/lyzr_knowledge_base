# Knowledge Graph Platform - Testing Guide

## Table of Contents
1. [Testing Overview](#testing-overview)
2. [Test Environment Setup](#test-environment-setup)
3. [Automated Testing](#automated-testing)
4. [Manual Testing](#manual-testing)
5. [Performance Testing](#performance-testing)
6. [Integration Testing](#integration-testing)
7. [API Testing](#api-testing)
8. [Troubleshooting Tests](#troubleshooting-tests)

---

## Testing Overview

The Knowledge Graph Platform includes comprehensive testing capabilities covering unit tests, integration tests, and manual testing procedures. The testing framework ensures system reliability, performance, and functionality validation.

### Test Categories

| Test Type | Coverage | Status | Requirements |
|-----------|----------|--------|--------------|
| **Unit Tests** | Core functionality | ✅ PASSING | Local environment |
| **Integration Tests** | End-to-end pipeline | ⚠️ REQUIRES API KEY | OpenAI API key |
| **API Tests** | REST endpoints | ✅ PASSING | Docker services |
| **WebSocket Tests** | Streaming functionality | ✅ PASSING | Docker services |
| **Performance Tests** | Load and stress | 📋 MANUAL | Production environment |

---

## Test Environment Setup

### Prerequisites

#### Required Software
- Docker and Docker Compose
- Python 3.11+
- Git
- Valid OpenAI API key (for integration tests)

### Environment Configuration

#### 1. Clone Repository
```bash
git clone <repository-url>
cd lyzr-knowledge-base
```

#### 2. Create Environment File
```bash
# Create .env file for testing
cat > .env << 'EOF'
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSION=3072

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=My_Password@123

# Vector Store Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=knowledge_graph

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Processing Configuration
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=200
ENTITY_RESOLUTION_THRESHOLD=0.85
DEDUP_THRESHOLD=0.95

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
EOF
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Automated Testing

### Running Test Suite

#### Start Infrastructure
```bash
# Start all services
docker compose -f docker/docker-compose.yml up -d

# Verify services are running
docker compose -f docker/docker-compose.yml ps
```

#### Run All Tests
```bash
# Run complete test suite
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ -v --cov=services --cov=api
```

#### Run Specific Test Categories
```bash
# Unit tests only
python -m pytest tests/unit/ -v

# Integration tests only
python -m pytest tests/integration/ -v

# Specific test file
python -m pytest tests/unit/test_entity_resolver.py -v
```

### Test Results Interpretation

#### Successful Test Run
```
============================= test session starts ==============================
platform darwin -- Python 3.11.9, pytest-7.4.3, pluggy-0.12
collecting ... collected 2 items

tests/unit/test_entity_resolver.py::test_entity_resolution PASSED        [ 50%]
tests/integration/test_document_processing.py::test_full_document_pipeline PASSED [100%]

============================== 2 passed in 15.23s ==============================
```

#### Test Failure Analysis
```
FAILED tests/integration/test_document_processing.py::test_full_document_pipeline
openai.AuthenticationError: Error code: 401 - Incorrect API key provided
```

**Common Failure Causes:**
- Missing or invalid OpenAI API key
- Insufficient API quota
- Service connectivity issues
- Missing environment variables

---

## Manual Testing

### Health Check Testing

#### 1. Basic Health Check
```bash
curl -s http://localhost:8000/health/ | jq .
```

**Expected Response:**
```json
{
  "status": "healthy",
  "graph_db": "connected",
  "vector_store": "connected"
}
```

#### 2. System Statistics
```bash
curl -s http://localhost:8000/health/stats | jq .
```

**Expected Response:**
```json
{
  "nodes": 0,
  "relationships": 0,
  "status": "operational"
}
```

### Document Processing Testing

#### 1. Ontology Generation
```bash
curl -X POST "http://localhost:8000/ontology/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "document_text": "Apple Inc. is a technology company founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. The company is headquartered in Cupertino, California and is known for its iPhone, iPad, and Mac products. Tim Cook is the current CEO of Apple."
  }' | jq .
```

**Expected Response:**
```json
{
  "success": true,
  "ontology": {
    "version": "1.0",
    "entity_types": [
      {
        "name": "Person",
        "description": "Individual people",
        "attributes": ["name", "role", "age"]
      },
      {
        "name": "Organization",
        "description": "Companies and institutions",
        "attributes": ["name", "location", "industry"]
      }
    ],
    "relationship_types": [
      {
        "name": "FOUNDED_BY",
        "description": "Person founded organization",
        "source": "Person",
        "target": "Organization"
      }
    ],
    "hierarchies": [],
    "attributes": {},
    "constraints": []
  }
}
```

#### 2. Document Processing
```bash
curl -X POST "http://localhost:8000/documents/process" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Albert Einstein was a theoretical physicist who developed the theory of relativity. He worked at Princeton University and won the Nobel Prize in Physics in 1921."
  }' | jq .
```

**Expected Response:**
```json
{
  "success": true,
  "stats": {
    "entities_created": 5,
    "relationships_created": 3,
    "chunks_processed": 1
  },
  "message": "Document processed successfully"
}
```

#### 3. File Upload Testing
```bash
# Create test document
echo "Tesla Inc. is an electric vehicle and clean energy company founded by Elon Musk. The company is headquartered in Austin, Texas and is known for its Model S, Model 3, and Model Y vehicles." > test_document.txt

# Upload and process
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@test_document.txt" | jq .
```

### Retrieval System Testing

#### 1. Knowledge Graph Query
```bash
curl -X POST "http://localhost:8000/retrieval/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What companies are mentioned in the documents?"
  }' | jq .
```

**Expected Response:**
```json
{
  "success": true,
  "answer": "Based on the processed documents, the following companies are mentioned: Apple Inc., Tesla Inc.",
  "sources": [
    {
      "entity": "Apple Inc.",
      "type": "Organization",
      "confidence": 0.95
    },
    {
      "entity": "Tesla Inc.",
      "type": "Organization", 
      "confidence": 0.92
    }
  ],
  "reasoning_chain": [
    {
      "type": "plan",
      "content": "Search for organizations in the knowledge graph"
    },
    {
      "type": "step",
      "content": "Executing graph traversal to find organization entities"
    }
  ]
}
```

#### 2. WebSocket Streaming Test
```javascript
// Create WebSocket connection
const ws = new WebSocket('ws://localhost:8000/retrieval/stream');

ws.onopen = function() {
    console.log('WebSocket connected');
    
    // Send query
    ws.send(JSON.stringify({
        "query": "Who founded Apple Inc.?"
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
    
    if (data.type === 'final_answer') {
        console.log('Final Answer:', data.content);
        ws.close();
    }
};

ws.onerror = function(error) {
    console.error('WebSocket error:', error);
};
```

---

## Performance Testing

### Load Testing

#### 1. API Endpoint Load Test
```bash
# Install Apache Bench
# On macOS: brew install httpd
# On Ubuntu: sudo apt-get install apache2-utils

# Test health endpoint
ab -n 1000 -c 10 http://localhost:8000/health/

# Test ontology generation (requires API key)
ab -n 100 -c 5 -p ontology_request.json -T "application/json" http://localhost:8000/ontology/generate
```

#### 2. Concurrent Document Processing
```bash
# Create multiple test documents
for i in {1..10}; do
  echo "Document $i: This is a test document about company $i." > test_doc_$i.txt
done

# Process documents concurrently
for i in {1..10}; do
  curl -X POST "http://localhost:8000/documents/upload" \
    -F "file=@test_doc_$i.txt" &
done
wait
```

### Memory and CPU Monitoring

#### 1. Docker Resource Monitoring
```bash
# Monitor container resources
docker stats

# Monitor specific service
docker stats docker-api-1
```

#### 2. Database Performance
```bash
# Neo4j performance metrics
curl -u neo4j:My_Password@123 http://localhost:7474/db/data/

# Qdrant metrics
curl http://localhost:6333/metrics
```

---

## Integration Testing

### End-to-End Pipeline Testing

#### 1. Complete Document Processing Pipeline
```bash
#!/bin/bash
# Complete pipeline test script

echo "Starting end-to-end pipeline test..."

# Step 1: Health check
echo "1. Checking system health..."
curl -s http://localhost:8000/health/ | jq .

# Step 2: Process document
echo "2. Processing test document..."
RESPONSE=$(curl -s -X POST "http://localhost:8000/documents/process" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Microsoft Corporation is a multinational technology company founded by Bill Gates and Paul Allen. The company is headquartered in Redmond, Washington and is known for its Windows operating system and Office productivity suite. Satya Nadella is the current CEO of Microsoft."
  }')

echo "Processing response: $RESPONSE"

# Step 3: Query the knowledge graph
echo "3. Querying knowledge graph..."
QUERY_RESPONSE=$(curl -s -X POST "http://localhost:8000/retrieval/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who is the CEO of Microsoft?"
  }')

echo "Query response: $QUERY_RESPONSE"

# Step 4: Check system stats
echo "4. Checking system statistics..."
curl -s http://localhost:8000/health/stats | jq .

echo "End-to-end pipeline test completed!"
```

#### 2. Multi-Document Processing Test
```bash
#!/bin/bash
# Multi-document processing test

echo "Starting multi-document processing test..."

# Create test documents
cat > doc1.txt << 'EOF'
Google LLC is a multinational technology company founded by Larry Page and Sergey Brin. The company is headquartered in Mountain View, California and is known for its search engine and Android operating system. Sundar Pichai is the current CEO of Google.
EOF

cat > doc2.txt << 'EOF'
Amazon.com Inc. is an e-commerce and cloud computing company founded by Jeff Bezos. The company is headquartered in Seattle, Washington and is known for its online marketplace and AWS cloud services. Andy Jassy is the current CEO of Amazon.
EOF

# Process documents
echo "Processing document 1..."
curl -X POST "http://localhost:8000/documents/upload" -F "file=@doc1.txt"

echo "Processing document 2..."
curl -X POST "http://localhost:8000/documents/upload" -F "file=@doc2.txt"

# Query across all documents
echo "Querying across all documents..."
curl -X POST "http://localhost:8000/retrieval/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "List all technology companies and their CEOs"
  }' | jq .

echo "Multi-document processing test completed!"
```

---

## API Testing

### REST API Testing

#### 1. API Endpoint Validation
```bash
#!/bin/bash
# API endpoint validation script

BASE_URL="http://localhost:8000"

echo "Testing API endpoints..."

# Test health endpoint
echo "Testing health endpoint..."
curl -s "$BASE_URL/health/" | jq .

# Test stats endpoint
echo "Testing stats endpoint..."
curl -s "$BASE_URL/health/stats" | jq .

# Test ontology generation
echo "Testing ontology generation..."
curl -s -X POST "$BASE_URL/ontology/generate" \
  -H "Content-Type: application/json" \
  -d '{"document_text": "Test document"}' | jq .

# Test document processing
echo "Testing document processing..."
curl -s -X POST "$BASE_URL/documents/process" \
  -H "Content-Type: application/json" \
  -d '{"text": "Test document"}' | jq .

# Test retrieval query
echo "Testing retrieval query..."
curl -s -X POST "$BASE_URL/retrieval/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Test query"}' | jq .

echo "API endpoint validation completed!"
```

#### 2. Error Handling Testing
```bash
#!/bin/bash
# Error handling test script

BASE_URL="http://localhost:8000"

echo "Testing error handling..."

# Test invalid endpoint
echo "Testing invalid endpoint..."
curl -s "$BASE_URL/invalid-endpoint" | jq .

# Test invalid JSON
echo "Testing invalid JSON..."
curl -s -X POST "$BASE_URL/ontology/generate" \
  -H "Content-Type: application/json" \
  -d 'invalid json' | jq .

# Test missing parameters
echo "Testing missing parameters..."
curl -s -X POST "$BASE_URL/ontology/generate" \
  -H "Content-Type: application/json" \
  -d '{}' | jq .

echo "Error handling test completed!"
```

### WebSocket Testing

#### 1. WebSocket Connection Test
```bash
#!/bin/bash
# WebSocket connection test using wscat

# Install wscat: npm install -g wscat

echo "Testing WebSocket connection..."

# Test WebSocket connection
wscat -c ws://localhost:8000/retrieval/stream

# Send test message
echo '{"query": "Test query"}' | wscat -c ws://localhost:8000/retrieval/stream
```

#### 2. WebSocket Streaming Test
```javascript
// WebSocket streaming test script
const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:8000/retrieval/stream');

let messageCount = 0;

ws.on('open', function open() {
    console.log('WebSocket connected');
    
    // Send test query
    ws.send(JSON.stringify({
        query: "What companies are mentioned in the knowledge graph?"
    }));
});

ws.on('message', function message(data) {
    messageCount++;
    const parsed = JSON.parse(data);
    console.log(`Message ${messageCount}:`, parsed);
    
    if (parsed.type === 'final_answer') {
        console.log('Streaming completed');
        ws.close();
    }
});

ws.on('error', function error(err) {
    console.error('WebSocket error:', err);
});

ws.on('close', function close() {
    console.log('WebSocket connection closed');
});
```

---

## Troubleshooting Tests

### Common Test Issues

#### 1. OpenAI API Issues
**Problem**: `AuthenticationError: Incorrect API key provided`

**Solutions**:
```bash
# Check API key in environment
echo $OPENAI_API_KEY

# Verify API key format
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models

# Test with minimal request
curl -X POST "https://api.openai.com/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 5
  }'
```

#### 2. Database Connection Issues
**Problem**: `Neo4j connection failed`

**Solutions**:
```bash
# Check Neo4j service status
docker compose -f docker/docker-compose.yml ps neo4j

# Check Neo4j logs
docker compose -f docker/docker-compose.yml logs neo4j

# Test Neo4j connection
docker compose -f docker/docker-compose.yml exec neo4j cypher-shell -u neo4j -p My_Password@123 "RETURN 1"
```

#### 3. Service Startup Issues
**Problem**: Services not starting properly

**Solutions**:
```bash
# Clean restart
docker compose -f docker/docker-compose.yml down -v
docker compose -f docker/docker-compose.yml up -d --build

# Check service logs
docker compose -f docker/docker-compose.yml logs -f

# Check resource usage
docker stats
```

### Test Environment Validation

#### 1. Environment Check Script
```bash
#!/bin/bash
# Environment validation script

echo "Validating test environment..."

# Check Docker
if command -v docker &> /dev/null; then
    echo "✅ Docker is installed"
    docker --version
else
    echo "❌ Docker is not installed"
    exit 1
fi

# Check Docker Compose
if command -v docker-compose &> /dev/null; then
    echo "✅ Docker Compose is installed"
    docker-compose --version
else
    echo "❌ Docker Compose is not installed"
    exit 1
fi

# Check Python
if command -v python3 &> /dev/null; then
    echo "✅ Python 3 is installed"
    python3 --version
else
    echo "❌ Python 3 is not installed"
    exit 1
fi

# Check pip
if command -v pip3 &> /dev/null; then
    echo "✅ pip3 is installed"
    pip3 --version
else
    echo "❌ pip3 is not installed"
    exit 1
fi

# Check .env file
if [ -f ".env" ]; then
    echo "✅ .env file exists"
else
    echo "❌ .env file is missing"
    exit 1
fi

# Check required environment variables
if [ -n "$OPENAI_API_KEY" ]; then
    echo "✅ OPENAI_API_KEY is set"
else
    echo "❌ OPENAI_API_KEY is not set"
fi

if [ -n "$NEO4J_PASSWORD" ]; then
    echo "✅ NEO4J_PASSWORD is set"
else
    echo "❌ NEO4J_PASSWORD is not set"
fi

echo "Environment validation completed!"
```

#### 2. Service Health Check Script
```bash
#!/bin/bash
# Service health check script

echo "Checking service health..."

# Check API service
echo "Checking API service..."
if curl -s http://localhost:8000/health/ > /dev/null; then
    echo "✅ API service is healthy"
else
    echo "❌ API service is not responding"
fi

# Check Neo4j service
echo "Checking Neo4j service..."
if curl -s http://localhost:7474 > /dev/null; then
    echo "✅ Neo4j service is healthy"
else
    echo "❌ Neo4j service is not responding"
fi

# Check Qdrant service
echo "Checking Qdrant service..."
if curl -s http://localhost:6333/collections > /dev/null; then
    echo "✅ Qdrant service is healthy"
else
    echo "❌ Qdrant service is not responding"
fi

# Check Redis service
echo "Checking Redis service..."
if docker compose -f docker/docker-compose.yml exec redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis service is healthy"
else
    echo "❌ Redis service is not responding"
fi

echo "Service health check completed!"
```

---

## Test Automation

### Continuous Integration Setup

#### 1. GitHub Actions Workflow
```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Start services
      run: |
        docker compose -f docker/docker-compose.yml up -d
        sleep 30
    
    - name: Run unit tests
      run: |
        python -m pytest tests/unit/ -v
    
    - name: Run integration tests
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        python -m pytest tests/integration/ -v
    
    - name: Stop services
      if: always()
      run: |
        docker compose -f docker/docker-compose.yml down
```

#### 2. Test Script Automation
```bash
#!/bin/bash
# Automated test runner

echo "Starting automated test suite..."

# Set test environment
export TEST_ENV=true
export LOG_LEVEL=INFO

# Start services
echo "Starting test services..."
docker compose -f docker/docker-compose.yml up -d
sleep 30

# Run test suite
echo "Running test suite..."
python -m pytest tests/ -v --junitxml=test-results.xml

# Generate coverage report
echo "Generating coverage report..."
python -m pytest tests/ --cov=services --cov=api --cov-report=html --cov-report=xml

# Run manual tests
echo "Running manual tests..."
bash scripts/manual_tests.sh

# Cleanup
echo "Cleaning up..."
docker compose -f docker/docker-compose.yml down

echo "Automated test suite completed!"
```

---

## Test Reporting

### Test Results Summary

#### 1. Test Coverage Report
```bash
# Generate HTML coverage report
python -m pytest tests/ --cov=services --cov=api --cov-report=html

# View coverage report
open htmlcov/index.html
```

#### 2. Performance Test Results
```bash
# Run performance tests
python -m pytest tests/performance/ -v --benchmark-only

# Generate performance report
python -m pytest tests/performance/ --benchmark-save=performance_results
```

#### 3. Test Documentation
```bash
# Generate test documentation
python -m pytest tests/ --collect-only > test_documentation.txt

# Generate API test report
python scripts/api_test_report.py > api_test_report.html
```

---

## Conclusion

This comprehensive testing guide provides all the necessary tools and procedures to validate the Knowledge Graph Platform's functionality, performance, and reliability. The testing framework ensures:

- **Functionality Validation**: All features work as expected
- **Performance Assurance**: System meets performance requirements
- **Reliability Testing**: System handles errors gracefully
- **Integration Verification**: All components work together seamlessly

### Key Testing Achievements
- ✅ **Automated Test Suite**: Unit and integration tests
- ✅ **Manual Testing Procedures**: Comprehensive API testing
- ✅ **Performance Testing**: Load and stress testing capabilities
- ✅ **Error Handling**: Robust error detection and reporting
- ✅ **Documentation**: Complete testing procedures and troubleshooting

The testing framework provides confidence in the platform's production readiness and ensures reliable operation across all use cases.

---

*Testing Guide Version: 1.0*  
*Last Updated: October 2025*  
*Platform Version: 1.0.0*
