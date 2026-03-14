"""
Semantic Scholar API search tool for academic papers.
"""
import requests
import os
from typing import List, Dict, Any


class SemanticScholarSearch:
    """
    Search client for Semantic Scholar API.
    """
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def _build_paper_list(self, data_items: list) -> List[Dict[str, Any]]:
        """
        Build a list of paper dictionaries from Semantic Scholar API response.
        
        Args:
            data_items: List of paper objects from API response
            
        Returns:
            List of formatted paper dictionaries
        """
        papers = []
        for item in data_items:
            authors = [a.get("name") for a in item.get("authors", [])]
            papers.append({
                "title": item.get("title"),
                "authors": authors,
                "year": item.get("year"),
                "abstract": item.get("abstract"),
                "venue": item.get("venue"),
                "citation_count": item.get("citationCount"),
                "is_open_access": item.get("isOpenAccess"),
                "source": "semantic_scholar",
                "link": item.get("url")
            })
        return papers

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search Semantic Scholar for papers.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of paper dictionaries with metadata
        """
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        headers = {}
        if api_key:
            headers["x-api-key"] = api_key

        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,year,abstract,venue,url,citationCount,isOpenAccess"
        }

        try:
            resp = requests.get(self.BASE_URL, params=params, headers=headers, timeout=10)
            if resp.status_code == 429:
                print("⚠️ Semantic Scholar: Rate limit exceeded (429).")
                return [{"error": "rate_limit",
                        "message": "Semantic Scholar API rate limit reached. Please wait."}]
            
            if resp.status_code == 403:
                print("⚠️ Semantic Scholar: Access Forbidden (403).")
                return [{"error": "forbidden",
                        "message": "Semantic Scholar access forbidden. Try again later."}]

            resp.raise_for_status()
            data = resp.json()
            return self._build_paper_list(data.get("data", []))
        except requests.exceptions.Timeout:
            print("⚠️ Semantic Scholar: Search timed out.")
            return [{"error": "timeout", "message": "Semantic Scholar search timed out."}]
        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"⚠️ Semantic Scholar Error: {e}")
            return []
