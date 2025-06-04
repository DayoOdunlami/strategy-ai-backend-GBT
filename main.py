# main.py - Main FastAPI Application (WITH FEEDBACK SYSTEM)
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from typing import List, Optional
import uvicorn
from datetime import datetime
import uuid
import os
import io
import csv
from pathlib import Path

# Import our modules
from database import DatabaseManager
from document_processor import EnhancedDocumentProcessor
from vector_store import PineconeManager
from web_scraper import ComprehensiveWebScraper
from ai_services import AIService
from auth import AdminAuth
from models import *  # All Pydantic models (now includes feedback models)
from config import settings

# Import enhanced modules
from specialized_agents import OrchestrationAgent
from report_generator import ReportGenerator, ReportTemplateManager
from map_data_manager import RailwayMapDataManager

# Initialize FastAPI app
app = FastAPI(
    title="Strategy AI Backend",
    description="AI Agent for Strategy Document Analysis with Admin Management & User Feedback",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for V0 frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize all services
db_manager = DatabaseManager()
doc_processor = EnhancedDocumentProcessor()
pinecone_manager = PineconeManager()
web_scraper = ComprehensiveWebScraper()
ai_service = AIService()
admin_auth = AdminAuth()

# Initialize enhanced services
orchestration_agent = OrchestrationAgent()
report_generator = ReportGenerator()
template_manager = ReportTemplateManager()
map_data_manager = RailwayMapDataManager()

# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Strategy AI Backend v2.0 is running!",
        "status": "healthy",
        "version": "2.0.0",
        "features": ["specialized_agents", "railway_maps", "report_generation", "user_feedback"],
        "timestamp": datetime.now()
    }

@app.get("/health")
async def health_check():
    """Detailed health check for monitoring"""
    try:
        db_status = await db_manager.test_connection()
        pinecone_status = await pinecone_manager.test_connection()
        
        return {
            "status": "healthy" if db_status and pinecone_status else "degraded",
            "services": {
                "database": "connected" if db_status else "error",
                "vector_store": "connected" if pinecone_status else "error",
                "document_processor": "ready",
                "web_scraper": "ready",
                "ai_service": "ready",
                "feedback_system": "enabled"
            },
            "metrics": {
                "total_documents": await db_manager.get_document_count(),
                "total_sectors": await db_manager.get_sector_count(),
                "total_use_cases": await db_manager.get_use_case_count(),
                "total_feedback": await db_manager.get_feedback_count()
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# ============================================================================
# MAIN CHAT ENDPOINT (For V0 Frontend)
# ============================================================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_ai(message: ChatMessage):
    """Main chat endpoint for V0 frontend"""
    try:
        # Auto-detect use case if not specified
        if not message.use_case:
            message.use_case = await ai_service.detect_use_case(
                message.message, message.sector
            )

        # Get custom prompt template
        prompt_template = await db_manager.get_prompt_template(
            message.sector, message.use_case
        )

        # Retrieve relevant documents
        relevant_docs = await pinecone_manager.semantic_search(
            query=message.message,
            filters={"sector": message.sector, "use_case": message.use_case},
            top_k=8
        )

        # Generate AI response
        ai_response = await ai_service.generate_response(
            query=message.message,
            context_docs=relevant_docs,
            prompt_template=prompt_template,
            user_type=message.user_type
        )

        # Format sources
        formatted_sources = [
            {
                "document_title": doc.get("metadata", {}).get("title", "Untitled"),
                "source": doc.get("metadata", {}).get("source", "Unknown"),
                "relevance_score": doc.get("score", 0.0),
                "chunk_preview": doc.get("text", "")[:200] + "..."
            }
            for doc in relevant_docs[:5]
        ]

        # Log interaction and get chat_log_id for feedback
        chat_log_id = await db_manager.log_chat_interaction(
            session_id=message.session_id,
            query=message.message,
            response=ai_response,
            sources_used=[doc.get("id") for doc in relevant_docs],
            sector=message.sector,
            use_case=message.use_case
        )

        return ChatResponse(
            response=ai_response,
            sources=formatted_sources,
            confidence=ai_service.calculate_confidence(relevant_docs),
            suggested_use_case=message.use_case,
            timestamp=datetime.now(),
            chat_log_id=chat_log_id  # Include for feedback linking
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/chat/advanced", response_model=ChatResponse)
async def advanced_chat_with_ai(message: ChatMessage):
    """
    Advanced chat endpoint using specialized AI agents
    Provides better responses through agent orchestration
    """
    try:
        # Determine complexity and agent requirements
        complexity = "comprehensive" if len(message.message) > 100 else "simple"
        
        # Retrieve relevant documents from vector store
        relevant_docs = await pinecone_manager.semantic_search(
            query=message.message,
            filters={"sector": message.sector, "use_case": message.use_case},
            top_k=8
        )

        # Prepare request for orchestration agent
        agent_request = {
            "type": "chat" if complexity == "simple" else "analysis",
            "query": message.message,
            "context": _prepare_context_for_agents(relevant_docs),
            "sector": message.sector,
            "use_case": message.use_case,
            "user_type": message.user_type,
            "complexity": complexity,
            "conversational_format": True
        }

        # Process with specialized agents
        agent_response = await orchestration_agent.process_request(agent_request)

        # Extract primary response
        if "primary_response" in agent_response:
            ai_response = agent_response["primary_response"]
            agents_used = agent_response.get("agents_used", [])
        else:
            ai_response = agent_response.get("response", "I'm having trouble processing your request.")
            agents_used = [agent_response.get("agent", "Unknown")]

        # Format sources for frontend
        formatted_sources = [
            {
                "document_title": doc.get("metadata", {}).get("title", "Untitled"),
                "source": doc.get("metadata", {}).get("source", "Unknown"),
                "relevance_score": doc.get("score", 0.0),
                "chunk_preview": doc.get("text", "")[:200] + "..."
            }
            for doc in relevant_docs[:5]
        ]

        # Log the enhanced interaction
        chat_log_id = await db_manager.log_chat_interaction(
            session_id=message.session_id,
            query=message.message,
            response=ai_response,
            sources_used=[doc.get("id") for doc in relevant_docs],
            sector=message.sector,
            use_case=message.use_case
        )

        return ChatResponse(
            response=ai_response,
            sources=formatted_sources,
            confidence=agent_response.get("confidence", 0.8),
            suggested_use_case=message.use_case,
            timestamp=datetime.now(),
            chat_log_id=chat_log_id,  # Include for feedback linking
            enhanced_features={
                "agents_used": agents_used,
                "response_type": agent_response.get("response_type", "standard"),
                "analysis_available": "detailed_analysis" in agent_response
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in advanced chat: {str(e)}")

def _prepare_context_for_agents(docs: List[Dict]) -> str:
    """Prepare context string optimized for AI agents"""
    if not docs:
        return "No relevant documents found."
    
    context_parts = []
    for i, doc in enumerate(docs[:5]):
        doc_title = doc.get("metadata", {}).get("title", f"Document {i+1}")
        doc_text = doc.get("text", "")
        doc_sector = doc.get("metadata", {}).get("sector", "")
        
        context_parts.append(f"Source {i+1} - {doc_title} ({doc_sector}):\n{doc_text}\n")
    
    return "\n".join(context_parts)

@app.get("/api/chat/suggest-use-case")
async def suggest_use_case(query: str, sector: str = "General"):
    """
    Conversational guidance feature - suggests best use case from user description
    For the hybrid dropdown + conversational interface
    """
    try:
        suggested_use_case = await ai_service.detect_use_case(query, sector)
        use_case_info = await db_manager.get_use_case_info(sector, suggested_use_case)
        
        return {
            "suggested_use_case": suggested_use_case,
            "confidence": 0.8,
            "description": use_case_info.get("description", ""),
            "example_queries": use_case_info.get("example_queries", []),
            "explanation": f"Based on your query, '{suggested_use_case}' seems most relevant for {sector} sector"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error suggesting use case: {str(e)}")

# ============================================================================
# USER FEEDBACK ENDPOINTS (NEW)
# ============================================================================

@app.post("/api/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: UserFeedback, request: Request):
    """Submit user feedback on AI responses or documents"""
    try:
        feedback_id = str(uuid.uuid4())
        
        # Get client info for analytics
        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        feedback_data = {
            "id": feedback_id,
            "chat_log_id": feedback.chat_log_id,
            "document_id": feedback.document_id,
            "session_id": feedback.session_id,
            "rating": feedback.rating,
            "feedback_type": feedback.feedback_type,
            "comment": feedback.comment,
            "helpful": feedback.helpful,
            "user_agent": user_agent,
            "ip_address": client_host,
            "created_at": datetime.utcnow().isoformat()
        }
        
        await db_manager.store_feedback(feedback_data)
        
        return FeedbackResponse(
            success=True,
            feedback_id=feedback_id,
            message="Thank you for your feedback! Your input helps us improve."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing feedback: {str(e)}")

@app.get("/api/feedback/analytics", response_model=FeedbackAnalytics)
async def get_feedback_analytics(
    days: int = 30,
    feedback_type: Optional[str] = None,
    admin_key: str = Depends(admin_auth.verify_admin)
):
    """Get feedback analytics for admin dashboard"""
    try:
        analytics = await db_manager.get_feedback_analytics(days=days, feedback_type=feedback_type)
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting feedback analytics: {str(e)}")

@app.get("/api/feedback/recent")
async def get_recent_feedback(
    limit: int = 20,
    admin_key: str = Depends(admin_auth.verify_admin)
):
    """Get recent feedback for monitoring"""
    try:
        recent_feedback = await db_manager.get_recent_feedback(limit=limit)
        return {"recent_feedback": recent_feedback}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recent feedback: {str(e)}")

# ============================================================================
# DOCUMENT MANAGEMENT ENDPOINTS (Enhanced with Feedback)
# ============================================================================

@app.post("/api/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    sector: str = Form("General"),
    use_case: Optional[str] = Form(None),
    tags: Optional[str] = Form(""),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200)
):
    """Upload and process documents with AI metadata generation"""
    try:
        # Validate file type
        allowed_types = {".pdf", ".docx", ".txt", ".csv", ".md"}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_ext}"
            )

        doc_id = str(uuid.uuid4())

        # Save file temporarily
        temp_file_path = f"/tmp/{doc_id}_{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)

        # Process in background (like your Streamlit processing)
        background_tasks.add_task(
            process_uploaded_document,
            doc_id=doc_id,
            file_path=temp_file_path,
            original_filename=file.filename,
            title=title,
            sector=sector,
            use_case=use_case,
            tags=tags,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        return {
            "success": True,
            "document_id": doc_id,
            "filename": file.filename,
            "message": "Document uploaded successfully. Processing in background...",
            "status_endpoint": f"/api/documents/{doc_id}/status"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.get("/api/documents", response_model=DocumentListResponse)
async def list_documents(
    sector: Optional[str] = None,
    use_case: Optional[str] = None,
    source_type: Optional[str] = None,
    search: Optional[str] = None,
    min_rating: Optional[float] = None,  # New: Filter by minimum rating
    limit: int = 50,
    offset: int = 0
):
    """List documents with filtering (enhanced with feedback data)"""
    try:
        documents = await db_manager.get_documents(
            sector=sector,
            use_case=use_case,
            source_type=source_type,
            search=search,
            min_rating=min_rating,
            limit=limit,
            offset=offset
        )

        total_count = await db_manager.get_document_count(
            sector=sector,
            use_case=use_case,
            source_type=source_type,
            search=search,
            min_rating=min_rating
        )

        return DocumentListResponse(
            documents=documents,
            total_count=total_count,
            limit=limit,
            offset=offset,
            has_more=(offset + limit) < total_count
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

# ============================================================================
# REPORT GENERATION ENDPOINTS
# ============================================================================

@app.post("/api/reports/generate")
async def generate_report(
    background_tasks: BackgroundTasks,
    report_type: str = Form(...),
    sector: str = Form("General"),
    format: str = Form("pdf"),  # pdf, docx, both
    title: Optional[str] = Form(None),
    scope: str = Form("comprehensive"),
    admin_key: str = Depends(admin_auth.verify_admin)
):
    """
    Generate comprehensive reports using AI agents
    Supports PDF and DOCX formats with professional styling
    """
    try:
        # Validate report type
        available_types = [rt["type"] for rt in report_generator.get_available_report_types()]
        if report_type not in available_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid report type. Available: {available_types}"
            )

        # Prepare report parameters
        parameters = {
            "sector": sector,
            "title": title,
            "scope": scope,
            "generated_by": "admin",
            "request_timestamp": datetime.now().isoformat()
        }

        # Start report generation in background
        report_task = report_generator.generate_report(
            report_type=report_type,
            parameters=parameters,
            format=format
        )

        # Execute the report generation
        result = await report_task

        if result["success"]:
            return {
                "success": True,
                "report_id": result["report_id"],
                "download_urls": result["download_urls"],
                "metadata": result["metadata"],
                "message": f"Report generated successfully in {format} format"
            }
        else:
            raise HTTPException(status_code=500, detail=result["error"])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

@app.get("/api/reports/{report_id}/download/{filename}")
async def download_report(report_id: str, filename: str):
    """Download generated report file"""
    try:
        file_path = await report_generator.get_report_file(report_id, filename)
        
        if not file_path or not Path(file_path).exists():
            raise HTTPException(status_code=404, detail="Report file not found")

        # Determine content type
        if filename.endswith('.pdf'):
            media_type = 'application/pdf'
        elif filename.endswith('.docx'):
            media_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        else:
            media_type = 'application/octet-stream'

        return FileResponse(
            path=file_path,
            filename=filename,
            media_type=media_type
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading report: {str(e)}")

# ============================================================================
# RAILWAY MAP DATA ENDPOINTS
# ============================================================================

@app.get("/api/map/regions")
async def get_railway_regions():
    """Get railway regions GeoJSON data for map visualization"""
    try:
        regions_data = await map_data_manager.get_railway_regions_geojson()
        return regions_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting railway regions: {str(e)}")

@app.get("/api/map/lines")
async def get_railway_lines():
    """Get railway lines GeoJSON data"""
    try:
        lines_data = await map_data_manager.get_railway_lines_geojson()
        return lines_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting railway lines: {str(e)}")

@app.get("/api/map/stations")
async def get_railway_stations(region: Optional[str] = None):
    """Get railway stations GeoJSON data, optionally filtered by region"""
    try:
        stations_data = await map_data_manager.get_stations_geojson(region=region)
        return stations_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting railway stations: {str(e)}")

# ============================================================================
# ADMIN ANALYTICS ENDPOINTS (Enhanced with Feedback)
# ============================================================================

@app.get("/api/admin/analytics", response_model=EnhancedSystemAnalytics)
async def get_enhanced_system_analytics(admin_key: str = Depends(admin_auth.verify_admin)):
    """Get comprehensive system analytics including feedback data"""
    try:
        # Get base analytics
        base_analytics = await db_manager.get_system_analytics()
        
        # Get feedback analytics
        feedback_analytics = await db_manager.get_feedback_analytics()
        
        # Combine into enhanced analytics
        enhanced_analytics = EnhancedSystemAnalytics(
            **base_analytics,
            feedback_summary=feedback_analytics
        )
        
        return enhanced_analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting analytics: {str(e)}")

# ============================================================================
# BACKGROUND PROCESSING FUNCTIONS
# ============================================================================

async def process_uploaded_document(
    doc_id: str, file_path: str, original_filename: str,
    title: Optional[str], sector: str, use_case: Optional[str],
    tags: Optional[str], chunk_size: int, chunk_overlap: int
):
    """Background processing for uploaded documents"""
    try:
        await db_manager.update_document_status(doc_id, "processing")

        # Process document (like your document_processor.py)
        chunks = await doc_processor.process_file(
            file_path=file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Generate AI metadata for each chunk (like your auto_tagger.py)
        for i, chunk in enumerate(chunks):
            metadata = await ai_service.generate_chunk_metadata(
                text=chunk["text"],
                sector=sector,
                use_case=use_case,
                suggested_title=title or original_filename
            )
            chunk["metadata"] = metadata

        # Store in database (like your vector_db.py)
        await db_manager.create_document(
            document_id=doc_id, title=title or original_filename,
            filename=original_filename, sector=sector, use_case=use_case,
            tags=tags, chunks=chunks, source_type="file"
        )

        # Store in Pinecone
        await pinecone_manager.store_document_chunks(doc_id, chunks)

        await db_manager.update_document_status(doc_id, "completed")

        # Clean up
        if os.path.exists(file_path):
            os.remove(file_path)

    except Exception as e:
        await db_manager.update_document_status(doc_id, "failed", error=str(e))
        if os.path.exists(file_path):
            os.remove(file_path)

# ============================================================================
# DEVELOPMENT SERVER
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
@app.middleware("http")
async def add_request_id_header(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response
