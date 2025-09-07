#!/usr/bin/env python3
"""
PDF Chat Web Application

A modern web interface for the PDF chat system with local PDF rendering.
Built with FastAPI backend and serves a React frontend with PDF.js integration.

Features:
- Interactive chat with PDF collections
- Local PDF viewing with highlighting
- Collection management
- Real-time chat interface
- Mobile-responsive design
"""

import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import existing chat functionality
from pdf_chat import PDFChatSession
from query_collection import CollectionQueryTool

app = FastAPI(title="PDF Chat Interface", version="1.0.0")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
chat_sessions: Dict[str, PDFChatSession] = {}
PDF_DIR = Path("pdfs")
CHROMA_DB_DIR = Path("chroma_db")

# Global ChromaDB client to avoid multiple instances
chroma_client = None

# Pydantic models for API
class ChatMessage(BaseModel):
    message: str
    collection: str
    pdf_filename: Optional[str] = None
    toc_sections: Optional[List[str]] = None  # Changed from toc_section to support multiple
    toc_section: Optional[str] = None  # Keep for backward compatibility
    page_range: Optional[str] = None
    model: Optional[str] = "gpt-5"
    verbosity: Optional[str] = "medium"
    synthesis: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    context_info: Dict[str, Any]
    timestamp: str

class CollectionInfo(BaseModel):
    name: str
    pdf_count: int
    total_chunks: int
    last_updated: Optional[str]

class PDFInfo(BaseModel):
    filename: str
    title: Optional[str]
    page_count: int
    file_size: int
    toc_sections: List[Dict[str, Any]]

# WebSocket manager for real-time chat
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

def find_pdf_path(pdf_filename: str, collection_name: str = None) -> Optional[Path]:
    """
    Find the actual path to a PDF file, considering the nested collection structure.
    Collection names are lowercase in ChromaDB but directories may be capitalized.
    
    Args:
        pdf_filename: Name of the PDF file
        collection_name: Lowercase collection name from ChromaDB
        
    Returns:
        Path to the PDF file if found, None otherwise
    """
    # If collection name is provided, try to find the corresponding directory
    if collection_name:
        # Check for exact match first (unlikely but possible)
        exact_path = PDF_DIR / collection_name / pdf_filename
        if exact_path.exists():
            return exact_path
        
        # Check for capitalized version (most common case)
        capitalized_name = collection_name.capitalize()
        capitalized_path = PDF_DIR / capitalized_name / pdf_filename
        if capitalized_path.exists():
            return capitalized_path
        
        # Check for all caps version
        uppercase_name = collection_name.upper()
        uppercase_path = PDF_DIR / uppercase_name / pdf_filename
        if uppercase_path.exists():
            return uppercase_path
    
    # Search all subdirectories in the pdfs folder as fallback
    for subdir in PDF_DIR.iterdir():
        if subdir.is_dir():
            pdf_path = subdir / pdf_filename
            if pdf_path.exists():
                return pdf_path
    
    # Final fallback: check if PDF is directly in pdfs/ (for backward compatibility)
    direct_path = PDF_DIR / pdf_filename
    if direct_path.exists():
        return direct_path
    
    return None

# API Routes
@app.get("/")
async def root():
    """Serve the main application page"""
    return FileResponse("static/index.html")

def get_chroma_client():
    """Get or create a shared ChromaDB client"""
    global chroma_client
    if chroma_client is None:
        import chromadb
        from chromadb.config import Settings
        chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
    return chroma_client

@app.get("/api/collections", response_model=List[CollectionInfo])
async def get_collections():
    """Get list of available collections"""
    collections = []
    
    # Use shared ChromaDB client to list collections
    client = get_chroma_client()
    chroma_collections = client.list_collections()
    
    for chroma_collection in chroma_collections:
        collection_name = chroma_collection.name
        
        # Get unique PDF filenames from metadata
        # Get all metadata from the collection
        results = chroma_collection.get(include=["metadatas"])
        unique_pdfs = set()
        
        for metadata in results.get("metadatas", []):
            pdf_filename = metadata.get("pdf_filename")
            if pdf_filename:
                unique_pdfs.add(pdf_filename)
        
        collections.append(CollectionInfo(
            name=collection_name,
            pdf_count=len(unique_pdfs),  # Actual unique PDF count
            total_chunks=chroma_collection.count(),  # Total chunks
            last_updated=None
        ))
                
    return collections

@app.get("/api/collections/{collection_name}/pdfs", response_model=List[PDFInfo])
async def get_collection_pdfs(collection_name: str):
    """Get PDFs in a specific collection"""
    try:
        # Use shared ChromaDB client to get collection data
        client = get_chroma_client()
        collection = client.get_collection(collection_name)
        
        # Get all metadata from the collection
        results = collection.get(include=["metadatas"])
        
        # Group by PDF filename and collect TOC sections
        pdf_data = {}
        for metadata in results.get("metadatas", []):
            pdf_filename = metadata.get("pdf_filename")
            if pdf_filename:
                if pdf_filename not in pdf_data:
                    pdf_data[pdf_filename] = {
                        "toc_sections": []
                    }
                
                # Collect TOC sections
                toc_title = metadata.get("toc_title")
                corrected_toc_page = metadata.get("corrected_toc_page", 0)
                if toc_title and toc_title not in [section.get("title") for section in pdf_data[pdf_filename]["toc_sections"]]:
                    pdf_data[pdf_filename]["toc_sections"].append({
                        "title": toc_title,
                        "page": corrected_toc_page
                    })
        
        # Build PDF list
        pdfs = []
        for pdf_name, data in pdf_data.items():
            # Use the helper function to find the actual PDF path
            pdf_path = find_pdf_path(pdf_name, collection_name)
            if pdf_path:
                # Sort TOC sections by page number
                toc_sections = sorted(data["toc_sections"], key=lambda x: x.get("page", 0))
                
                pdfs.append(PDFInfo(
                    filename=pdf_name,
                    title=pdf_name.replace(".pdf", "").replace("_", " "),
                    page_count=0,  # Could extract from metadata if available
                    file_size=pdf_path.stat().st_size,
                    toc_sections=toc_sections
                ))
        
        return pdfs
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found: {e}")

@app.get("/api/pdfs/{filename}")
async def serve_pdf(filename: str, collection: Optional[str] = None):
    """Serve PDF files from local storage, searching in nested directories"""
    # Use the helper function to find the actual PDF path
    pdf_path = find_pdf_path(filename, collection)
    
    if not pdf_path:
        raise HTTPException(status_code=404, detail=f"PDF '{filename}' not found")
    
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=filename
    )

@app.get("/api/collections/{collection_name}/pdfs/{filename}")
async def serve_collection_pdf(collection_name: str, filename: str):
    """Serve PDF files from a specific collection directory"""
    # Use the helper function with the specific collection name
    pdf_path = find_pdf_path(filename, collection_name)
    
    if not pdf_path:
        raise HTTPException(status_code=404, detail=f"PDF '{filename}' not found in collection '{collection_name}'")
    
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=filename
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatMessage):
    """Main chat endpoint"""
    try:
        # Get or create chat session
        session_key = f"{chat_request.collection}_{chat_request.pdf_filename or 'all'}"
        
        if session_key not in chat_sessions:
            chat_sessions[session_key] = PDFChatSession(
                collection_name=chat_request.collection,
                vector_store_dir=str(CHROMA_DB_DIR),
                enhanced_mode=False
            )
        
        session = chat_sessions[session_key]
        
        # Set chat mode based on request (order matters - most specific first)
        if chat_request.page_range and chat_request.pdf_filename:
            # Parse page range (assuming format like "1-10")
            try:
                start_page, end_page = map(int, chat_request.page_range.split("-"))
                session.set_chat_mode("page_range", 
                                    pdf_filename=chat_request.pdf_filename,
                                    page_range=(start_page, end_page))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid page range format")
        elif chat_request.pdf_filename and (chat_request.toc_sections or chat_request.toc_section):
            # Handle multiple TOC sections or single section (for backward compatibility)
            toc_sections = chat_request.toc_sections or [chat_request.toc_section]
            session.set_chat_mode("toc_section", 
                                pdf_filename=chat_request.pdf_filename, 
                                toc_sections=toc_sections)
        elif chat_request.pdf_filename:
            session.set_chat_mode("pdf", pdf_filename=chat_request.pdf_filename)
        else:
            session.set_chat_mode("collection")
        
        # Process the chat message
        response_text = session.ask_question(chat_request.message, model=chat_request.model, verbosity=chat_request.verbosity, synthesis=chat_request.synthesis)
        
        # Get context info for the response
        context_info = {
            "mode": session.current_context["mode"],
            "pdf": session.current_context.get("pdf_filenames"),
            "toc_sections": session.current_context.get("toc_sections"),
            "tokens_used": getattr(session, 'last_tokens_used', 0),
            "sources_count": len(getattr(session, 'last_sources', []))
        }
        
        # Extract sources if available
        sources = getattr(session, 'last_sources', [])
        
        return ChatResponse(
            response=response_text,
            sources=sources,
            context_info=context_info,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        import traceback
        print(f"ERROR in chat_endpoint: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat-history/{collection_name}")
async def get_chat_history(collection_name: str):
    """Get chat history for a specific collection."""
    try:
        # Create a temporary session just to access the history methods
        temp_session = PDFChatSession(
            collection_name=collection_name,
            vector_store_dir=str(CHROMA_DB_DIR),
            enhanced_mode=False
        )
        
        collection_messages = temp_session.get_chat_history_for_collection()
        return {"messages": collection_messages}
        
    except Exception as e:
        print(f"Error loading chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear-chat/{collection_name}")
async def clear_chat_history(collection_name: str):
    """Mark chat history as cleared for a specific collection."""
    try:
        # Create a temporary session just to access the history methods
        temp_session = PDFChatSession(
            collection_name=collection_name,
            vector_store_dir=str(CHROMA_DB_DIR),
            enhanced_mode=False
        )
        
        cleared_count = temp_session.clear_chat_history_for_collection()
        
        # IMPORTANT: Invalidate all cached sessions for this collection to force reload of conversation history
        sessions_to_remove = []
        for session_key in chat_sessions:
            if session_key.startswith(f"{collection_name}_"):
                sessions_to_remove.append(session_key)
        
        for session_key in sessions_to_remove:
            del chat_sessions[session_key]
            print(f"🗑️ Invalidated cached session: {session_key}")
        
        return {
            "success": True,
            "message": f"Chat history cleared for collection '{collection_name}'",
            "messages_cleared": cleared_count
        }
        
    except Exception as e:
        print(f"Error clearing chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chunks/{collection_name}/{chunk_id}")
async def get_chunk_content(collection_name: str, chunk_id: str):
    """Get chunk content by chunk ID for highlighting"""
    try:
        # Initialize a query tool to access the vector store
        query_tool = CollectionQueryTool(
            collection_name=collection_name, 
            vector_store_dir=str(CHROMA_DB_DIR)
        )
        
        # Query the collection for the specific chunk
        results = query_tool.vector_store.collection.get(ids=[chunk_id])
        
        if not results["documents"] or len(results["documents"]) == 0:
            raise HTTPException(status_code=404, detail="Chunk not found")
        
        # Extract chunk data
        chunk_content = results["documents"][0]
        chunk_metadata = results["metadatas"][0] if results["metadatas"] else {}
        
        return {
            "chunk_id": chunk_id,
            "content": chunk_content,
            "metadata": chunk_metadata
        }
        
    except Exception as e:
        print(f"Error retrieving chunk {chunk_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving chunk: {str(e)}")

@app.websocket("/api/ws/{collection_name}")
async def websocket_chat(websocket: WebSocket, collection_name: str):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(websocket)
    
    # Initialize chat session
    session = PDFChatSession(collection_name, vector_store_dir=str(CHROMA_DB_DIR))
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Process the message
            message = data.get("message", "")
            pdf_filename = data.get("pdf_filename")
            toc_sections = data.get("toc_sections", [])
            toc_section = data.get("toc_section")  # For backward compatibility
            # Set appropriate mode
            if pdf_filename and (toc_sections or toc_section):
                # Handle multiple TOC sections or single section (for backward compatibility)
                final_toc_sections = toc_sections or [toc_section]
                session.set_chat_mode("toc_section", 
                                    pdf_filename=pdf_filename, 
                                    toc_sections=final_toc_sections)
            elif pdf_filename:
                session.set_chat_mode("pdf", pdf_filename=pdf_filename)
            else:
                session.set_chat_mode("collection")
            
            # Process and send response (enhanced mode permanently disabled)
            response = session.process_message(message, enhanced=False)
            
            await manager.send_personal_message(json.dumps({
                "response": response,
                "timestamp": datetime.now().isoformat(),
                "context": {
                    "mode": session.current_context["mode"],
                    "pdf": session.current_context.get("pdf_filenames"),
                    "toc_sections": session.current_context.get("toc_sections")
                }
            }), websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Static file serving for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    
    print("🚀 Starting PDF Chat Web Application...")
    print(f"📁 PDF Directory: {PDF_DIR.absolute()}")
    print(f"🗄️  ChromaDB Directory: {CHROMA_DB_DIR.absolute()}")
    print("🌐 Access at: http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)