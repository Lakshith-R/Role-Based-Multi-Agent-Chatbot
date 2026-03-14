"""
Utility functions for paper search aggregation and deduplication.
"""
from typing import List, Dict


def normalize_papers(*source_lists: List[Dict]) -> List[Dict]:
    """
    Merge and deduplicate paper results based on title from multiple sources.
    """
    seen_titles = set()
    merged = []

    for source_list in source_lists:
        if not source_list:
            continue
        for paper in source_list:
            if not isinstance(paper, dict) or "title" not in paper:
                continue
            
            title = paper.get("title", "")
            if not title:
                continue
                
            title_key = title.lower().strip()
            # Basic cleanup for matching
            title_key = "".join(e for e in title_key if e.isalnum())
            
            if title_key in seen_titles:
                continue
                
            seen_titles.add(title_key)
            merged.append(paper)

    return merged
