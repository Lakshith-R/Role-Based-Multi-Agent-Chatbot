"""
Google Books API search tool for book enrichment and fallback.
"""
import requests
import os
from typing import List, Dict, Any


class GoogleBooksSearch:
    """
    Search client for Google Books API (Enrichment & Fallback).
    """
    BASE_URL = "https://www.googleapis.com/books/v1/volumes"

    def _build_book_list(self, items: list) -> List[Dict[str, Any]]:
        """
        Build a list of book dictionaries from Google Books API response.
        
        Args:
            items: List of item objects from API response
            
        Returns:
            List of formatted book dictionaries
        """
        books = []
        for item in items:
            info = item.get("volumeInfo", {})
            books.append({
                "title": info.get("title"),
                "authors": info.get("authors", []),
                "publisher": info.get("publisher"),
                "year": info.get("publishedDate")[:4] if info.get("publishedDate") else None,
                "categories": info.get("categories", []),
                "page_count": info.get("pageCount"),
                "source": "google_books",
                "link": info.get("infoLink", "")
            })
        return books

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search Google Books API.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of book dictionaries with metadata
        """
        api_key = os.getenv("GOOGLE_BOOKS_API_KEY")  # Optional for public data
        
        params = {
            "q": query,
            "printType": "books",
            "langRestrict": "en",
            "maxResults": limit
        }
        if api_key:
            params["key"] = api_key

        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return self._build_book_list(data.get("items", []))
        except Exception as e: # pylint: disable=broad-exception-caught
            return [{"error": str(e)}]
