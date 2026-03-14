"""
Open Library API search tool for academic books.
"""
import requests
from typing import List, Dict, Any


class OpenLibrarySearch:
    """
    Search client for Open Library API (Academic focus).
    """
    BASE_URL = "https://openlibrary.org/search.json"

    def _build_book_list(self, docs: list, limit: int) -> List[Dict[str, Any]]:
        """
        Build a list of book dictionaries from Open Library API response.
        
        Args:
            docs: List of document objects from API response
            limit: Maximum number of books to process
            
        Returns:
            List of formatted book dictionaries
        """
        books = []
        for doc in docs[:limit]:
            books.append({
                "title": doc.get("title"),
                "authors": doc.get("author_name", []),
                "year": doc.get("first_publish_year"),
                "subjects": doc.get("subject", [])[:6],
                "publisher": doc.get("publisher", [])[:3] if doc.get("publisher") else [],
                "edition_count": doc.get("edition_count", 0),
                "source": "openlibrary",
                "link": f"https://openlibrary.org{doc.get('key')}" if doc.get('key') else ""
            })
        return books

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search Open Library for books.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of book dictionaries with metadata
        """
        params = {
            "q": query,
            "language": "eng",
            "has_fulltext": "true",  # String "true" often works better with some APIs
            "limit": limit
        }

        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return self._build_book_list(data.get("docs", []), limit)
        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"OpenLibrary API Error: {e}")
            return []
