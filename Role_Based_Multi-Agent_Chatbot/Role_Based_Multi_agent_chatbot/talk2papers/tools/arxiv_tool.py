"""
ArXiv API search tool for CS and AI research papers.
"""
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any


class ArXivSearch:
    """
    Search client for ArXiv API (CS and AI focused).
    """
    BASE_URL = "http://export.arxiv.org/api/query"

    def _build_paper_list(self, entries: list, namespace: dict) -> List[Dict[str, Any]]:
        """
        Build a list of paper dictionaries from ArXiv API response.
        
        Args:
            entries: List of entry elements from XML response
            namespace: XML namespace dictionary
            
        Returns:
            List of formatted paper dictionaries
        """


        papers = []
        for entry in entries:
            title = entry.find('atom:title', namespace).text.strip().replace('\n', ' ')
            abstract = entry.find('atom:summary', namespace).text.strip().replace('\n', ' ')
            year = entry.find('atom:published', namespace).text[:4]
            link = entry.find('atom:id', namespace).text
            
            authors_found = entry.findall('atom:author', namespace)
            authors = [a.find('atom:name', namespace).text for a in authors_found]
            
            papers.append({
                "title": title,
                "authors": authors,
                "year": year,
                "abstract": abstract,
                "link": link,
                "source": "arxiv"
            })
        return papers

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search ArXiv for papers.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of paper dictionaries with metadata
        """
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }

        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            
            # ArXiv returns XML (Atom feed)
            root = ET.fromstring(resp.text)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            entries = root.findall('atom:entry', ns)
            return self._build_paper_list(entries, ns)
        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"⚠️ ArXiv Search Error: {e}")
            return []
