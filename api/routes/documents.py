from fastapi import APIRouter, UploadFile, File, HTTPException
from api.models.schemas import ProcessDocumentRequest, ProcessDocumentResponse
from services.ingestion.document_processor import DocumentProcessor
import api.main as main_app

router = APIRouter()

@router.post("/process", response_model=ProcessDocumentResponse)
async def process_document(request: ProcessDocumentRequest):
    """Process document and build knowledge graph"""
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
    """Upload and process document file"""
    try:
        content = await file.read()
        text = content.decode('utf-8')
        
        processor = DocumentProcessor(main_app.graph_db)
        stats = await processor.process_document(text)
        
        return {"success": True, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
