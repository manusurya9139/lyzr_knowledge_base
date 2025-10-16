from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from api.models.schemas import QueryRequest
from services.retrieval.agent_orchestrator import AgentOrchestrator
import api.main as main_app
import json

router = APIRouter()

@router.websocket("/stream")
async def retrieval_stream(websocket: WebSocket):
    """Streaming retrieval with reasoning chain"""
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
    """Non-streaming retrieval endpoint"""
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
