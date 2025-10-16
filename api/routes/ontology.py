from fastapi import APIRouter, HTTPException
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
    """Generate ontology from document"""
    try:
        generator = OntologyGenerator()
        ontology = await generator.generate_ontology(request.document_text)
        return {"success": True, "ontology": ontology}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/refine")
async def refine_ontology(request: RefineOntologyRequest):
    """Refine ontology based on feedback"""
    try:
        generator = OntologyGenerator()
        refined = await generator.refine_ontology(request.ontology, request.feedback)
        return {"success": True, "ontology": refined}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/merge")
async def merge_ontologies(request: MergeOntologiesRequest):
    """Merge multiple ontologies"""
    try:
        generator = OntologyGenerator()
        merged = await generator.merge_ontologies(request.ontologies)
        return {"success": True, "ontology": merged}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
