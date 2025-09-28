import os
import uuid
import logging
import ssl
import urllib3
from typing import List, Dict, Any, Optional
from pathlib import Path
import aiofiles
import PyPDF2
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from config import config

# Disable SSL warnings and verification for corporate networks
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# Set environment variables for HuggingFace Hub to disable SSL verification
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['SSL_VERIFY'] = 'false'

logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger('chromadb.telemetry.posthog').setLevel(logging.ERROR)

class DocumentProcessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )
        
        # Initialize ChromaDB with telemetry disabled
        self.chroma_client = chromadb.PersistentClient(
            path=config.CHROMA_PERSIST_DIRECTORY,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.get_collection("documents")
        except:
            self.collection = self.chroma_client.create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
    
    async def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from different file types"""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return await self._extract_from_pdf(file_path)
        elif file_extension == '.docx':
            return await self._extract_from_docx(file_path)
        elif file_extension in ['.txt', '.md']:
            return await self._extract_from_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    async def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    async def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    async def _extract_from_text(self, file_path: str) -> str:
        """Extract text from text files"""
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
            return await file.read()
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        return self.text_splitter.split_text(text)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks"""
        return self.embedding_model.encode(texts).tolist()
    
    async def process_and_store_document(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Process document and store in vector database"""
        try:
            # Extract text
            text = await self.extract_text_from_file(file_path)
            
            if not text.strip():
                raise ValueError("No text content found in the document")
            
            # Chunk text
            chunks = self.chunk_text(text)
            
            if not chunks:
                raise ValueError("No chunks generated from the document")
            
            # Generate embeddings
            embeddings = self.generate_embeddings(chunks)
            
            # Generate unique IDs for chunks
            document_id = str(uuid.uuid4())
            chunk_ids = [f"{document_id}_{i}" for i in range(len(chunks))]
            
            # Prepare metadata
            metadatas = [
                {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_path": file_path
                }
                for i in range(len(chunks))
            ]
            
            # Store in ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=chunk_ids
            )
            
            return {
                "document_id": document_id,
                "filename": filename,
                "total_chunks": len(chunks),
                "status": "success"
            }
            
        except Exception as e:
            return {
                "filename": filename,
                "status": "error",
                "error": str(e)
            }
    
    def search_documents(self, query: str, n_results: int = None) -> Dict[str, Any]:
        """Search documents using semantic similarity"""
        if n_results is None:
            n_results = config.TOP_K_DOCUMENTS
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else []
        }
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about stored documents"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "status": "success"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_all_documents(self) -> Dict[str, Any]:
        """Get list of all documents in the knowledge base"""
        try:
            # Get all documents with metadata
            results = self.collection.get(include=["metadatas"])
            
            # Group by document
            documents = {}
            for metadata in results["metadatas"]:
                doc_id = metadata["document_id"]
                filename = metadata["filename"]
                
                if doc_id not in documents:
                    documents[doc_id] = {
                        "document_id": doc_id,
                        "filename": filename,
                        "total_chunks": metadata["total_chunks"],
                        "file_path": metadata.get("file_path", ""),
                        "chunks": []
                    }
                
                documents[doc_id]["chunks"].append({
                    "chunk_index": metadata["chunk_index"],
                    "total_chunks": metadata["total_chunks"]
                })
            
            # Convert to list and sort by filename
            document_list = list(documents.values())
            document_list.sort(key=lambda x: x["filename"])
            
            return {
                "documents": document_list,
                "total_documents": len(document_list),
                "total_chunks": len(results["metadatas"]),
                "status": "success"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "documents": []
            }
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete a document and all its chunks from the knowledge base"""
        try:
            # Get all chunks for this document
            results = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )
            
            if not results["ids"]:
                return {
                    "status": "error",
                    "error": "Document not found"
                }
            
            # Get document info before deletion
            filename = results["metadatas"][0]["filename"] if results["metadatas"] else "Unknown"
            chunk_count = len(results["ids"])
            
            # Delete all chunks for this document
            self.collection.delete(where={"document_id": document_id})
            
            # Try to delete the physical file if it exists
            file_path = results["metadatas"][0].get("file_path", "") if results["metadatas"] else ""
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass  # File might be in use or already deleted
            
            return {
                "status": "success",
                "message": f"Document '{filename}' deleted successfully",
                "deleted_chunks": chunk_count,
                "filename": filename
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            } 