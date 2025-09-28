import json
import httpx
from typing import Dict, List, Any, Optional
from document_processor import DocumentProcessor
from config import config

class RAGPipeline:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.api_base = config.OPENAI_API_BASE
        self.api_key = config.OPENAI_API_KEY
        self.api_version = config.OPENAI_API_VERSION
        self.deployment_name = config.OPENAI_DEPLOYMENT_NAME
        
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
    
    async def generate_answer(self, query: str) -> Dict[str, Any]:
        """Generate answer using RAG pipeline with completeness detection"""
        
        # Step 1: Retrieve relevant documents
        search_results = self.document_processor.search_documents(query)
        
        if not search_results["documents"]:
            return {
                "answer": "I don't have any relevant documents to answer your question.",
                "confidence": 0.0,
                "missing_info": ["No relevant documents found in the knowledge base"],
                "enrichment_suggestions": [
                    "Upload documents related to your query",
                    "Try rephrasing your question with different keywords"
                ],
                "sources": []
            }
        
        # Step 2: Prepare context from retrieved documents
        context = self._prepare_context(search_results)
        
        # Step 3: Generate answer with completeness analysis
        response = await self._call_openai_with_completeness_check(query, context)
        
        # Step 4: Calculate confidence based on retrieval scores
        confidence = self._calculate_confidence(search_results["distances"])
        
        # Step 5: Prepare sources information
        sources = self._prepare_sources(search_results)
        
        return {
            "answer": response.get("answer", ""),
            "confidence": confidence,
            "missing_info": response.get("missing_info", []),
            "enrichment_suggestions": response.get("enrichment_suggestions", []),
            "sources": sources
        }
    
    def _prepare_context(self, search_results: Dict[str, Any]) -> str:
        """Prepare context from retrieved documents"""
        documents = search_results["documents"]
        metadatas = search_results["metadatas"]
        
        context_parts = []
        for i, doc in enumerate(documents):
            metadata = metadatas[i]
            context_parts.append(
                f"Document: {metadata['filename']}\n"
                f"Content: {doc}\n"
                f"---"
            )
        
        return "\n".join(context_parts)
    
    async def _call_openai_with_completeness_check(self, query: str, context: str) -> Dict[str, Any]:
        """Call OpenAI API with structured prompt for completeness analysis"""
        
        system_prompt = """You are an AI assistant that answers questions based on provided documents and analyzes the completeness of your answers.

Your task is to:
1. Answer the user's question using ONLY the information from the provided documents
2. Analyze if your answer is complete or if important information is missing
3. Suggest specific ways to enrich the knowledge base if information is incomplete

Respond in the following JSON format:
{
    "answer": "Your detailed answer based on the documents",
    "missing_info": ["List of specific information that would make the answer more complete"],
    "enrichment_suggestions": ["Specific suggestions for documents, data, or sources to add"]
}

CRITICAL Guidelines:
- NEVER include specific facts, dates, names, or details in missing_info or enrichment_suggestions that are not present in the provided documents
- Only describe TYPES of information that are missing (e.g., "company history", "technical specifications") without providing the actual details
- LIMIT missing_info to maximum 2 items, keep them short and generic
- LIMIT enrichment_suggestions to maximum 2 items, keep them concise
- If the documents don't contain information about the topic, simply state that information about the topic is missing
- Do NOT hallucinate or provide factual details from your training data
- Be brief and generic about what additional information would be helpful
- If the answer is complete, set missing_info to an empty array

Examples of CORRECT missing_info (max 2 items):
- "Basic company information"
- "Product details"

Examples of CORRECT enrichment_suggestions (max 2 items):
- "Add company overview documents"
- "Include product information"

Examples of INCORRECT (too detailed, too many items):
- "Historical background and founding details, Details about products and achievements, Information about impact on industry, Market presence data"
"""

        user_prompt = f"""Context from documents:
{context}

Question: {query}

Please provide a comprehensive answer and analysis."""

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 1500,
            "temperature": 0.3,
            "response_format": {"type": "json_object"}
        }
        
        try:
            # Create an httpx client with SSL verification disabled and proper timeout
            timeout = httpx.Timeout(60.0)
            with httpx.Client(verify=False, timeout=timeout) as client:
                # Make the POST request to the Azure OpenAI API
                response = client.post(
                    f'{self.api_base}/openai/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}',
                    headers=self.headers, 
                    data=json.dumps(payload)
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    return json.loads(content)
                else:
                    return {
                        "answer": f"Error calling OpenAI API: {response.status_code} - {response.text}",
                        "missing_info": ["API call failed"],
                        "enrichment_suggestions": ["Check API configuration and try again"]
                    }
                    
        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "missing_info": ["System error occurred"],
                "enrichment_suggestions": ["Check system configuration and try again"]
            }
    
    def _calculate_confidence(self, distances: List[float]) -> float:
        """Calculate confidence score based on retrieval distances"""
        if not distances:
            return 0.0
        
        # Convert distances to similarity scores (lower distance = higher similarity)
        # Assuming cosine distance, convert to confidence (0-1 scale)
        avg_distance = sum(distances) / len(distances)
        
        # Convert distance to confidence (this is a heuristic)
        # Distances typically range from 0 to 2 for cosine similarity
        confidence = max(0.0, min(1.0, 1.0 - (avg_distance / 2.0)))
        
        return round(confidence, 3)
    
    def _prepare_sources(self, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare source information from search results"""
        sources = []
        metadatas = search_results["metadatas"]
        distances = search_results["distances"]
        
        for i, metadata in enumerate(metadatas):
            sources.append({
                "filename": metadata["filename"],
                "chunk_index": metadata["chunk_index"],
                "total_chunks": metadata["total_chunks"],
                "relevance_score": round(1.0 - (distances[i] / 2.0), 3) if i < len(distances) else 0.0
            })
        
        return sources
    
    async def suggest_auto_enrichment(self, missing_info: List[str], query: str) -> List[Dict[str, Any]]:
        """Generate suggestions for auto-enrichment from external sources"""
        if not missing_info:
            return []
        
        suggestions = []
        
        for info in missing_info:
            # Generate search suggestions for external sources
            suggestion = {
                "missing_topic": info,
                "suggested_sources": [
                    f"Search for '{info}' in academic papers",
                    f"Look up '{info}' in industry documentation",
                    f"Find recent articles about '{info}'"
                ],
                "search_queries": [
                    f"{info} {query}",
                    f"{info} best practices",
                    f"{info} latest research"
                ]
            }
            suggestions.append(suggestion)
        
        return suggestions
    
    async def rate_answer_quality(self, query: str, answer: str, rating: int, feedback: str = "") -> Dict[str, Any]:
        """Store user feedback for answer quality improvement"""
        # This would typically be stored in a database for model improvement
        # For now, we'll just return a confirmation
        
        feedback_data = {
            "query": query,
            "answer": answer,
            "rating": rating,
            "feedback": feedback,
            "timestamp": "2024-01-01T00:00:00Z"  # In real implementation, use actual timestamp
        }
        
        # In a production system, you would:
        # 1. Store this in a database
        # 2. Use it for model fine-tuning
        # 3. Analyze patterns in low-rated answers
        
        return {
            "status": "success",
            "message": "Thank you for your feedback! This will help improve future responses."
        } 