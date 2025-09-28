import httpx
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from config import config
from logger import weave_logger

class AutoEnrichment:
    def __init__(self):
        self.api_base = config.OPENAI_API_BASE
        self.api_key = config.OPENAI_API_KEY
        self.api_version = config.OPENAI_API_VERSION
        self.deployment_name = config.OPENAI_DEPLOYMENT_NAME
        
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
    
    async def generate_enrichment_content(self, user_query: str) -> Dict[str, Any]:
        """Generate enrichment content using OpenAI completion API"""
        weave_logger.log_info(f"🤖 Generating enrichment content for query: '{user_query}'")
        
        try:
            # Create a comprehensive prompt for enrichment
            enrichment_prompt = f"""You are a knowledgeable assistant. The user asked: "{user_query}"

Please provide comprehensive, factual information to answer this question. Include:
- Key facts and background information
- Important details and context
- Relevant examples or explanations
- Any additional useful information related to the topic

Write in a clear, informative style suitable for a knowledge base. Aim for 3-4 paragraphs of detailed information."""

            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides comprehensive, factual information on any topic."
                    },
                    {
                        "role": "user", 
                        "content": enrichment_prompt
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.3,
                "top_p": 0.9
            }
            
            weave_logger.log_info(f"🤖 Sending request to OpenAI API...")
            
            # Use a longer timeout for OpenAI API calls (60 seconds)
            timeout = httpx.Timeout(60.0)
            with httpx.Client(verify=False, timeout=timeout) as client:
                response = client.post(
                    f'{self.api_base}/openai/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}',
                    headers=self.headers,
                    data=json.dumps(payload)
                )
                
                weave_logger.log_info(f"🤖 OpenAI API Response: {response.json()}")
                
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    
                    weave_logger.log_info(f"🤖 Successfully generated enrichment content ({len(content)} characters)")
                    
                    return {
                        "query": user_query,
                        "content": content,
                        "status": "success",
                        "source": "OpenAI Knowledge Generation",
                        "generated_at": datetime.now().isoformat()
                    }
                else:
                    error_msg = f"OpenAI API error: {response.status_code}"
                    weave_logger.log_info(f"🤖 OpenAI API Error: {error_msg}")
                    return {
                        "query": user_query,
                        "status": "error",
                        "error": error_msg
                    }
                    
        except Exception as e:
            weave_logger.log_error(e, "generate_enrichment_content")
            return {
                "query": user_query,
                "status": "error",
                "error": str(e)
            }
    
    def _create_short_filename(self, user_query: str) -> str:
        """Create a short, meaningful filename from the user query"""
        import re
        
        # Remove common question words and clean up
        query_clean = user_query.lower()
        query_clean = re.sub(r'\b(what|who|where|when|why|how|is|are|was|were|do|does|did|can|could|would|should|tell|me|about|the|a|an)\b', '', query_clean)
        
        # Remove special characters and extra spaces
        query_clean = re.sub(r'[^\w\s]', '', query_clean)
        query_clean = re.sub(r'\s+', ' ', query_clean).strip()
        
        # Split into words and take the most meaningful ones
        words = [word for word in query_clean.split() if len(word) > 2]
        
        # Take first 2-3 most meaningful words
        if len(words) >= 3:
            key_words = words[:3]
        elif len(words) >= 2:
            key_words = words[:2]
        elif len(words) >= 1:
            key_words = words[:1]
        else:
            # Fallback to first few characters of original query
            key_words = [user_query.replace(' ', '')[:8].lower()]
        
        # Join with underscores and limit length
        short_name = '_'.join(key_words)[:20]  # Max 20 characters
        
        return short_name
    
    async def auto_enrich_knowledge_base(self, user_query: str, document_processor) -> Dict[str, Any]:
        """Automatically enrich the knowledge base with OpenAI-generated content"""
        weave_logger.log_info(f"🚀 Starting auto-enrichment for query: '{user_query}'")
        
        try:
            # Generate enrichment content using OpenAI
            enrichment_data = await self.generate_enrichment_content(user_query)
            
            if enrichment_data["status"] == "success":
                # Create enriched document content
                short_query = self._create_short_filename(user_query)
                temp_filename = f"auto_enriched_{short_query}.txt"
                temp_filepath = f"./uploads/{temp_filename}"
                
                # Build the enriched content
                enriched_content = f"""AUTO-ENRICHMENT: {user_query}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Source: OpenAI Knowledge Generation

{'='*80}

QUERY: {user_query}

ENRICHMENT CONTENT:
{enrichment_data['content']}

{'='*80}

This content was automatically generated to enrich the knowledge base in response to the query: "{user_query}"

Generated using OpenAI's knowledge synthesis capabilities.
"""
                
                # Write the enriched content to file
                with open(temp_filepath, 'w', encoding='utf-8') as f:
                    f.write(enriched_content)
                
                # Process and store in knowledge base
                result = await document_processor.process_and_store_document(
                    temp_filepath, 
                    temp_filename
                )
                
                if result["status"] == "success":
                    weave_logger.log_info(f"✅ Auto-enrichment successful: {temp_filename} ({result.get('total_chunks', 0)} chunks)")
                    return {
                        "status": "success",
                        "filename": temp_filename,
                        "chunks_added": result.get("total_chunks", 0),
                        "query": user_query,
                        "content_length": len(enrichment_data['content'])
                    }
                else:
                    weave_logger.log_info(f"❌ Failed to store enrichment: {result.get('error', 'Unknown error')}")
                    return {
                        "status": "error",
                        "error": f"Failed to store enrichment: {result.get('error', 'Unknown error')}"
                    }
            else:
                return {
                    "status": "error",
                    "error": enrichment_data.get("error", "Failed to generate enrichment content")
                }
                
        except Exception as e:
            weave_logger.log_error(e, "auto_enrich_knowledge_base")
            return {
                "status": "error",
                "error": str(e)
            }

# Create global instance
auto_enrichment = AutoEnrichment() 