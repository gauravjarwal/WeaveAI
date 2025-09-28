import httpx
import json
import asyncio
from typing import Dict, List, Any, Optional
from urllib.parse import quote
import xml.etree.ElementTree as ET
from datetime import datetime
from logger import weave_logger

class ExternalSourceFetcher:
    """Fetches data from trusted external sources for auto-enrichment"""
    
    def __init__(self):
        self.timeout = 30.0
        self.max_results_per_source = 3
        
    async def fetch_from_multiple_sources(self, query: str, missing_topics: List[str]) -> List[Dict[str, Any]]:
        """Fetch content from multiple trusted external sources"""
        all_results = []
        
        # Create search tasks for different sources
        tasks = []
        
        for topic in missing_topics[:3]:  # Limit to 3 topics to avoid rate limits
            # Wikipedia
            tasks.append(self._fetch_from_wikipedia(topic))
            # arXiv (for academic content)
            tasks.append(self._fetch_from_arxiv(topic))
            # Web search (using DuckDuckGo Instant Answer API)
            tasks.append(self._fetch_from_duckduckgo(topic))
        
        # Execute all tasks concurrently
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, dict) and result.get("status") == "success":
                    all_results.extend(result.get("sources", []))
                elif isinstance(result, Exception):
                    weave_logger.log_error(result, "external_source_fetch")
                    
        except Exception as e:
            weave_logger.log_error(e, "fetch_from_multiple_sources")
        
        # Sort by confidence and return top results
        all_results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return all_results[:5]  # Return top 5 results
    
    async def _fetch_from_wikipedia(self, topic: str) -> Dict[str, Any]:
        """Fetch content from Wikipedia API"""
        try:
            encoded_topic = quote(topic)
            
            # First, search for the topic
            search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_topic}"
            weave_logger.log_info(f"🌐 Wikipedia API Request: {search_url}")
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(search_url)
                weave_logger.log_info(f"🌐 Wikipedia API Response: Status {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    weave_logger.log_info(f"🌐 Wikipedia API Data: Title='{data.get('title', 'N/A')}', Extract Length={len(data.get('extract', ''))}")
                    
                    if data.get("extract"):
                        result = {
                            "status": "success",
                            "sources": [{
                                "title": data.get("title", topic),
                                "content": data.get("extract", ""),
                                "source": "Wikipedia",
                                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                                "confidence": 0.9,
                                "type": "encyclopedia",
                                "fetched_at": datetime.now().isoformat()
                            }]
                        }
                        weave_logger.log_info(f"🌐 Wikipedia SUCCESS: Found content for '{topic}'")
                        return result
                    else:
                        weave_logger.log_info(f"🌐 Wikipedia NO CONTENT: No extract found for '{topic}'")
                else:
                    weave_logger.log_info(f"🌐 Wikipedia FAILED: Status {response.status_code} for '{topic}'")
                        
        except Exception as e:
            weave_logger.log_error(e, f"wikipedia_fetch_{topic}")
            weave_logger.log_info(f"🌐 Wikipedia ERROR: {str(e)} for '{topic}'")
        
        return {"status": "failed", "sources": []}
    
    async def _fetch_from_arxiv(self, topic: str) -> Dict[str, Any]:
        """Fetch academic papers from arXiv API"""
        try:
            # arXiv API search
            encoded_topic = quote(topic)
            arxiv_url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_topic}&start=0&max_results=2"
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(arxiv_url)
                
                if response.status_code == 200:
                    # Parse XML response
                    root = ET.fromstring(response.content)
                    
                    sources = []
                    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                        title = entry.find('{http://www.w3.org/2005/Atom}title')
                        summary = entry.find('{http://www.w3.org/2005/Atom}summary')
                        link = entry.find('{http://www.w3.org/2005/Atom}id')
                        
                        if title is not None and summary is not None:
                            sources.append({
                                "title": title.text.strip(),
                                "content": summary.text.strip()[:1000] + "..." if len(summary.text) > 1000 else summary.text.strip(),
                                "source": "arXiv",
                                "url": link.text if link is not None else "",
                                "confidence": 0.8,
                                "type": "academic",
                                "fetched_at": datetime.now().isoformat()
                            })
                    
                    if sources:
                        return {"status": "success", "sources": sources}
                        
        except Exception as e:
            weave_logger.log_error(e, f"arxiv_fetch_{topic}")
        
        return {"status": "failed", "sources": []}
    
    async def _fetch_from_duckduckgo(self, topic: str) -> Dict[str, Any]:
        """Fetch instant answers from DuckDuckGo API"""
        try:
            encoded_topic = quote(topic)
            ddg_url = f"https://api.duckduckgo.com/?q={encoded_topic}&format=json&no_html=1&skip_disambig=1"
            weave_logger.log_info(f"🦆 DuckDuckGo API Request: {ddg_url}")
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(ddg_url)
                weave_logger.log_info(f"🦆 DuckDuckGo API Response: Status {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    weave_logger.log_info(f"🦆 DuckDuckGo API Data: Abstract={bool(data.get('Abstract'))}, Definition={bool(data.get('Definition'))}")
                    
                    sources = []
                    
                    # Check for instant answer
                    if data.get("Abstract"):
                        sources.append({
                            "title": data.get("Heading", topic),
                            "content": data.get("Abstract", ""),
                            "source": "DuckDuckGo Instant Answer",
                            "url": data.get("AbstractURL", ""),
                            "confidence": 0.7,
                            "type": "instant_answer",
                            "fetched_at": datetime.now().isoformat()
                        })
                        weave_logger.log_info(f"🦆 DuckDuckGo ABSTRACT: Found for '{topic}'")
                    
                    # Check for definition
                    if data.get("Definition"):
                        sources.append({
                            "title": f"Definition: {topic}",
                            "content": data.get("Definition", ""),
                            "source": data.get("DefinitionSource", "DuckDuckGo"),
                            "url": data.get("DefinitionURL", ""),
                            "confidence": 0.6,
                            "type": "definition",
                            "fetched_at": datetime.now().isoformat()
                        })
                        weave_logger.log_info(f"🦆 DuckDuckGo DEFINITION: Found for '{topic}'")
                    
                    if sources:
                        weave_logger.log_info(f"🦆 DuckDuckGo SUCCESS: {len(sources)} sources for '{topic}'")
                        return {"status": "success", "sources": sources}
                    else:
                        weave_logger.log_info(f"🦆 DuckDuckGo NO CONTENT: No abstract or definition for '{topic}'")
                else:
                    weave_logger.log_info(f"🦆 DuckDuckGo FAILED: Status {response.status_code} for '{topic}'")
                        
        except Exception as e:
            weave_logger.log_error(e, f"duckduckgo_fetch_{topic}")
            weave_logger.log_info(f"🦆 DuckDuckGo ERROR: {str(e)} for '{topic}'")
        
        return {"status": "failed", "sources": []}
    
    async def _fetch_from_news_api(self, topic: str, api_key: str = None) -> Dict[str, Any]:
        """Fetch recent news articles (requires API key)"""
        if not api_key:
            return {"status": "failed", "sources": []}
        
        try:
            encoded_topic = quote(topic)
            news_url = f"https://newsapi.org/v2/everything?q={encoded_topic}&sortBy=relevancy&pageSize=2&apiKey={api_key}"
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(news_url)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    sources = []
                    for article in data.get("articles", []):
                        if article.get("description"):
                            sources.append({
                                "title": article.get("title", ""),
                                "content": article.get("description", "") + "\n\n" + (article.get("content", "") or "")[:500],
                                "source": f"News: {article.get('source', {}).get('name', 'Unknown')}",
                                "url": article.get("url", ""),
                                "confidence": 0.6,
                                "type": "news",
                                "fetched_at": datetime.now().isoformat()
                            })
                    
                    if sources:
                        return {"status": "success", "sources": sources}
                        
        except Exception as e:
            weave_logger.log_error(e, f"news_api_fetch_{topic}")
        
        return {"status": "failed", "sources": []}
    
    async def _fetch_from_github(self, topic: str) -> Dict[str, Any]:
        """Fetch relevant repositories from GitHub API"""
        try:
            encoded_topic = quote(topic)
            github_url = f"https://api.github.com/search/repositories?q={encoded_topic}&sort=stars&order=desc&per_page=2"
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(github_url)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    sources = []
                    for repo in data.get("items", []):
                        if repo.get("description"):
                            sources.append({
                                "title": repo.get("full_name", ""),
                                "content": repo.get("description", "") + f"\n\nLanguage: {repo.get('language', 'N/A')}\nStars: {repo.get('stargazers_count', 0)}",
                                "source": "GitHub",
                                "url": repo.get("html_url", ""),
                                "confidence": 0.5,
                                "type": "code_repository",
                                "fetched_at": datetime.now().isoformat()
                            })
                    
                    if sources:
                        return {"status": "success", "sources": sources}
                        
        except Exception as e:
            weave_logger.log_error(e, f"github_fetch_{topic}")
        
        return {"status": "failed", "sources": []}

# Global instance
external_fetcher = ExternalSourceFetcher() 