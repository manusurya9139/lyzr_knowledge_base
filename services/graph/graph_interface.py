from abc import ABC, abstractmethod
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
    """Unified interface for graph databases"""
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to graph database"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection"""
        pass
    
    @abstractmethod
    async def create_node(self, node: Node) -> str:
        """Create a node and return its ID"""
        pass
    
    @abstractmethod
    async def create_relationship(self, rel: Relationship) -> str:
        """Create a relationship and return its ID"""
        pass
    
    @abstractmethod
    async def find_nodes(self, labels: List[str], properties: Dict[str, Any]) -> List[Node]:
        """Find nodes matching criteria"""
        pass
    
    @abstractmethod
    async def traverse(self, start_node_id: str, relationship_types: List[str], 
                      max_depth: int = 3) -> List[Dict[str, Any]]:
        """Traverse graph from starting node"""
        pass
    
    @abstractmethod
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute native query (Cypher/Gremlin)"""
        pass
