from pydantic import BaseModel, Field
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
