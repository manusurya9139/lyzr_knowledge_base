from fastapi import APIRouter
import api.main as main_app

router = APIRouter()

@router.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "graph_db": "connected" if main_app.graph_db else "disconnected",
        "vector_store": "connected" if main_app.embedding_store else "disconnected"
    }

@router.get("/stats")
async def get_stats():
    """Get system statistics"""
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
