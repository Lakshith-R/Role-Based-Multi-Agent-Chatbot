"""
CORE API search tool for open access academic papers.
"""
import requests
import os
from typing import List, Dict, Any


class CoreSearch:
    """
    Search client for CORE API (Open Access).
    """
    BASE_URL = "https://api.core.ac.uk/v3/search/works"

    def _build_paper_list(self, results: list) -> List[Dict[str, Any]]:
        """
        Build a list of paper dictionaries from CORE API response.
        
        Args:
            results: List of result objects from API response
            
        Returns:
            List of formatted paper dictionaries
        """
        papers = []
        for item in results:
            authors = [a.get("name") for a in item.get("authors", [])]
            papers.append({
                "title": item.get("title"),
                "authors": authors,
                "year": item.get("yearPublished"),
                "abstract": item.get("abstract"),
                "link": item.get("downloadUrl") or item.get("fullTextIdentifier"),
                "source": "core",
                "publisher": item.get("publisher")
            })
        return papers

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search CORE API.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of paper dictionaries with metadata
        """
        api_key = os.getenv("CORE_API_KEY")
        if not api_key:
            # CORE usually requires an API key for search
            print("⚠️ CORE API key missing. Skipping CORE search.")
            return []

        headers = {"Authorization": f"Bearer {api_key}"}
        params = {
            "q": query,
            "limit": limit
        }

        try:
            resp = requests.get(self.BASE_URL, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return self._build_paper_list(data.get("results", []))
        except requests.exceptions.Timeout:
            print("⚠️ CORE API: Search timed out.")
            return [{"error": "timeout", "message": "CORE search timed out."}]
        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"⚠️ CORE API Error: {e}")
            return []
