"""
Utility functions for book search aggregation and deduplication.
"""
from typing import List, Dict


def normalize_books(openlib_results: List[Dict], google_results: List[Dict]) -> List[Dict]:
    """
    Merge and deduplicate book results based on title.
    """
    seen_titles = set()
    merged = []

    # Prioritize OpenLibrary (Academic) then Google Books
    for book in openlib_results + google_results:
        title = book.get("title", "")
        if not title:
            continue
            
        # Create a normalized component for deduping
        title_key = title.lower().strip()
        
        # Simple fuzzy check (exact match on normalized title)
        if title_key in seen_titles:
            continue
            
        seen_titles.add(title_key)
        merged.append(book)

    return merged
