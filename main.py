import os
import shutil
import time
import ssl
import urllib3
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

# Disable SSL warnings and verification for corporate networks
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_VERIFY'] = 'false'

from config import config
from document_processor import DocumentProcessor
from rag_pipeline import RAGPipeline
from auto_enrichment import AutoEnrichment
from logger import weave_logger

# Create directories if they don't exist
os.makedirs(config.UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(config.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app = FastAPI(
    title="WeaveAI - Knowledge Base Search & Enrichment",
    description="AI-powered knowledge base with search, completeness detection, and enrichment suggestions",
    version="1.0.0"
)

# Initialize components
document_processor = DocumentProcessor()
rag_pipeline = RAGPipeline()
auto_enrichment = AutoEnrichment()

# Setup static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Pydantic models
class QueryRequest(BaseModel):
    query: str

class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: int
    feedback: Optional[str] = ""

class AutoEnrichRequest(BaseModel):
    user_query: str

class DeleteDocumentRequest(BaseModel):
    document_id: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main web interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process multiple documents"""
    results = []
    
    for file in files:
        # Validate file type
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in config.ALLOWED_EXTENSIONS:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": f"Unsupported file type: {file_extension}"
            })
            continue
        
        # Check file size
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        if file_size > config.MAX_FILE_SIZE_MB * 1024 * 1024:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": f"File too large. Maximum size: {config.MAX_FILE_SIZE_MB}MB"
            })
            continue
        
        # Save file
        file_path = os.path.join(config.UPLOAD_DIRECTORY, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        # Process document
        try:
            result = await document_processor.process_and_store_document(file_path, file.filename)
            results.append(result)
            # Log the upload
            weave_logger.log_document_upload(file.filename, result)
        except Exception as e:
            error_result = {
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            }
            results.append(error_result)
            weave_logger.log_document_upload(file.filename, error_result)
    
    return {"results": results}

@app.post("/search")
async def search_query(request: QueryRequest):
    """Search the knowledge base and generate an answer"""
    start_time = time.time()
    
    try:
        weave_logger.log_info(f"Processing query: '{request.query}'")
        
        result = await rag_pipeline.generate_answer(request.query)
        
        # Add auto-enrichment suggestions if there's missing info
        if result["missing_info"]:
            weave_logger.log_info(f"Missing info detected: {result['missing_info']}")
            auto_enrichment = await rag_pipeline.suggest_auto_enrichment(
                result["missing_info"], 
                request.query
            )
            result["auto_enrichment"] = auto_enrichment
            weave_logger.log_enrichment(result["missing_info"], {"suggestions": auto_enrichment})
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Log the complete query and response
        weave_logger.log_query(request.query, result, processing_time)
        
        return result
    except Exception as e:
        weave_logger.log_error(e, "search_query")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for answer quality"""
    try:
        result = await rag_pipeline.rate_answer_quality(
            request.query,
            request.answer,
            request.rating,
            request.feedback
        )
        
        # Log the feedback
        weave_logger.log_feedback(request.query, request.answer, request.rating, request.feedback)
        
        return result
    except Exception as e:
        weave_logger.log_error(e, "submit_feedback")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get knowledge base statistics"""
    try:
        stats = document_processor.get_document_stats()
        return stats
    except Exception as e:
        weave_logger.log_error(e, "get_stats")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def get_all_documents():
    """Get list of all documents in the knowledge base"""
    try:
        documents = document_processor.get_all_documents()
        return documents
    except Exception as e:
        weave_logger.log_error(e, "get_all_documents")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the knowledge base"""
    try:
        result = document_processor.delete_document(document_id)
        if result["status"] == "success":
            weave_logger.log_info(f"Document deleted: {result['filename']} ({result['deleted_chunks']} chunks)")
        return result
    except Exception as e:
        weave_logger.log_error(e, "delete_document")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feedbacks")
async def get_all_feedbacks():
    """Get all user feedbacks from the feedback log file"""
    try:
        import json
        from pathlib import Path
        
        feedback_file = Path("./logs/feedback.jsonl")
        feedbacks = []
        
        if feedback_file.exists():
            with open(feedback_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            feedback_data = json.loads(line)
                            # Format timestamp for display
                            from datetime import datetime
                            timestamp = datetime.fromisoformat(feedback_data["timestamp"])
                            feedback_data["formatted_timestamp"] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                            # Truncate long queries and answers for display
                            feedback_data["short_query"] = feedback_data["query"][:100] + "..." if len(feedback_data["query"]) > 100 else feedback_data["query"]
                            feedback_data["short_answer"] = feedback_data["answer"][:200] + "..." if len(feedback_data["answer"]) > 200 else feedback_data["answer"]
                            feedbacks.append(feedback_data)
                        except json.JSONDecodeError:
                            continue
        
        # Return most recent feedbacks first
        feedbacks.reverse()
        
        return {
            "status": "success",
            "feedbacks": feedbacks,
            "total_count": len(feedbacks)
        }
    except Exception as e:
        weave_logger.log_error(e, "get_all_feedbacks")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auto-enrich")
async def auto_enrich_knowledge_base(request: AutoEnrichRequest):
    """Automatically enrich the knowledge base with OpenAI-generated content"""
    try:
        weave_logger.log_info(f"Starting auto-enrichment for query: '{request.user_query}'")
        
        result = await auto_enrichment.auto_enrich_knowledge_base(
            request.user_query,
            document_processor
        )
        
        # Log the enrichment results
        weave_logger.log_info(f"Auto-enrichment completed: {result.get('status')} - {result.get('filename', 'N/A')}")
        
        return result
    except Exception as e:
        weave_logger.log_error(e, "auto_enrich_endpoint")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "WeaveAI"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=True
    ) 