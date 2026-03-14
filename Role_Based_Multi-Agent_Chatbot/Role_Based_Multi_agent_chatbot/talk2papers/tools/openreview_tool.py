"""
OpenReview API search tool for conference papers (ICLR, NeurIPS, etc.).
"""
import requests
from typing import List, Dict, Any


class OpenReviewSearch:
    """
    Search client for OpenReview API (ICLR, NeurIPS, etc.).
    Uses API v2 primary and API v1 as fallback.
    """
    V2_URL = "https://api2.openreview.net/notes/search"
    V1_URL = "https://api.openreview.net/notes/search"

    def _build_paper_list_v2(self, notes: list) -> List[Dict[str, Any]]:
        """
        Build a list of paper dictionaries from OpenReview API v2 response.
        
        Args:
            notes: List of note objects from API v2 response
            
        Returns:
            List of formatted paper dictionaries
        """
        papers = []
        for item in notes:
            content = item.get("content", {})
            # API v2 has content values nested under 'value' metadata
            title = content.get("title", {}).get("value")
            authors = content.get("authors", {}).get("value", [])
            abstract = content.get("abstract", {}).get("value")
            pdf = content.get("pdf", {}).get("value")
            
            papers.append({
                "title": title,
                "authors": authors if isinstance(authors, list) else [authors],
                "year": item.get("cdate") or item.get("tcdate"),  # Use creation date
                "abstract": abstract,
                "venue": item.get("invitation", "").split("/")[0],  # Rough venue
                "link": f"https://openreview.net/forum?id={item.get('id')}",
                "pdf_link": f"https://openreview.net{pdf}" if pdf else None,
                "source": "openreview_v2"
            })
        return papers

    def _build_paper_list_v1(self, notes: list) -> List[Dict[str, Any]]:
        """
        Build a list of paper dictionaries from OpenReview API v1 response.
        
        Args:
            notes: List of note objects from API v1 response
            
        Returns:
            List of formatted paper dictionaries
        """
        papers = []
        for item in notes:
            content = item.get("content", {})
            # API v1 has content values directly
            papers.append({
                "title": content.get("title"),
                "authors": content.get("authors", []),
                "year": item.get("cdate") or item.get("tcdate"),
                "abstract": content.get("abstract"),
                "venue": item.get("invitation", "").split("/")[0],
                "link": f"https://openreview.net/forum?id={item.get('id')}",
                "source": "openreview_v1"
            })
        return papers

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search OpenReview for papers.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of paper dictionaries with metadata
        """
        papers = []
        
        # Try API v2 (Newer conferences like ICLR 2024+)
        try:
            resp = requests.get(self.V2_URL, params={"term": query, "limit": limit},
                              timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                papers = self._build_paper_list_v2(data.get("notes", []))
        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"OpenReview API v2 Error: {e}")

        # If we got results from V2, return them. Otherwise try V1.
        if papers:
            return papers

        # Try API v1 (Older conferences)
        try:
            resp = requests.get(self.V1_URL, params={"term": query, "limit": limit},
                              timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                papers = self._build_paper_list_v1(data.get("notes", []))
        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"⚠️ OpenReview V1 Search Error: {e}")

        return papers
