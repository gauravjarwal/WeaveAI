import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

class WeaveAILogger:
    def __init__(self, log_dir: str = "./logs"):
        """Initialize the WeaveAI logging system"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup main logger
        self.logger = logging.getLogger("WeaveAI")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler for all logs
        file_handler = logging.FileHandler(self.log_dir / "weaveai.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(console_handler)
        
        # Query-specific log file
        self.query_log_file = self.log_dir / "queries.jsonl"
        
    def log_query(self, query: str, response: Dict[str, Any], processing_time: float = None):
        """Log a search query and its response"""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "query": query,
            "response": {
                "answer": response.get("answer", ""),
                "confidence": response.get("confidence", 0),
                "missing_info": response.get("missing_info", []),
                "enrichment_suggestions": response.get("enrichment_suggestions", []),
                "sources": response.get("sources", []),
                "auto_enrichment": response.get("auto_enrichment", [])
            },
            "processing_time_seconds": processing_time,
            "metadata": {
                "sources_count": len(response.get("sources", [])),
                "missing_info_count": len(response.get("missing_info", [])),
                "enrichment_suggestions_count": len(response.get("enrichment_suggestions", []))
            }
        }
        
        # Log to console
        self.logger.info(f"Query processed: '{query}' | Confidence: {response.get('confidence', 0):.2f} | Sources: {len(response.get('sources', []))}")
        
        # Log detailed JSON to file
        with open(self.query_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    def log_document_upload(self, filename: str, result: Dict[str, Any]):
        """Log document upload events"""
        timestamp = datetime.now().isoformat()
        
        if result.get("status") == "success":
            self.logger.info(f"Document uploaded successfully: {filename} | Chunks: {result.get('total_chunks', 0)}")
        else:
            self.logger.error(f"Document upload failed: {filename} | Error: {result.get('error', 'Unknown error')}")
        
        log_entry = {
            "timestamp": timestamp,
            "event": "document_upload",
            "filename": filename,
            "result": result
        }
        
        upload_log_file = self.log_dir / "uploads.jsonl"
        with open(upload_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    def log_enrichment(self, missing_topics: list, enrichment_result: Dict[str, Any]):
        """Log enrichment activities"""
        timestamp = datetime.now().isoformat()
        
        self.logger.info(f"Enrichment processed: {len(missing_topics)} topics | Success: {enrichment_result.get('successful_enrichments', 0)}")
        
        log_entry = {
            "timestamp": timestamp,
            "event": "auto_enrichment",
            "missing_topics": missing_topics,
            "result": enrichment_result
        }
        
        enrichment_log_file = self.log_dir / "enrichment.jsonl"
        with open(enrichment_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    def log_feedback(self, query: str, answer: str, rating: int, feedback: str = ""):
        """Log user feedback"""
        timestamp = datetime.now().isoformat()
        
        self.logger.info(f"Feedback received: Rating {rating}/5 for query: '{query[:50]}...'")
        
        log_entry = {
            "timestamp": timestamp,
            "event": "user_feedback",
            "query": query,
            "answer": answer,
            "rating": rating,
            "feedback": feedback
        }
        
        feedback_log_file = self.log_dir / "feedback.jsonl"
        with open(feedback_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log errors with context"""
        self.logger.error(f"Error in {context}: {str(error)}", exc_info=True)
    
    def log_info(self, message: str):
        """Log general information"""
        self.logger.info(message)
    
    def get_recent_queries(self, limit: int = 10) -> list:
        """Get recent queries from log"""
        if not self.query_log_file.exists():
            return []
        
        queries = []
        with open(self.query_log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[-limit:]:
                try:
                    queries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        return list(reversed(queries))

# Global logger instance
weave_logger = WeaveAILogger() 