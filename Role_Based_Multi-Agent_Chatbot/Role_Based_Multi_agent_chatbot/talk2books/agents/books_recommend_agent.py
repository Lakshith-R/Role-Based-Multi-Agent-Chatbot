"""
Books recommendation agent using Google Books and Open Library APIs.
"""
import json
from pathlib import Path
from dotenv import load_dotenv

from agentic_student_assistant.core.base.base_agent import BaseAgent
from agentic_student_assistant.core.utils.config_loader import get_config
from agentic_student_assistant.core.utils.prompt_loader import load_agent_prompts
from agentic_student_assistant.talk2books.tools.googlebooks_tool import GoogleBooksSearch
from agentic_student_assistant.talk2books.tools.openlibrary_tool import OpenLibrarySearch
from agentic_student_assistant.talk2books.tools.book_utils import normalize_books


class BooksRecommendAgent(BaseAgent):
    """
    Agent for recommending academic books using Google Books and Open Library APIs.
    """

    def __init__(self):
        """Initialize book recommendation agent."""
        config = get_config()
        super().__init__(config, agent_name="books")
        # Load local prompts
        agent_path = Path(__file__).parent.parent
        prompts = load_agent_prompts(agent_path)
        self.recommendation_prompt = prompts['books_recommendation_academic']
        self.google_search = GoogleBooksSearch()
        self.openlibrary_search = OpenLibrarySearch()

    def _refine_query(self, query: str) -> str:
        """
        Extract core search terms from natural language query.
        Example: "recommend books on machine learning" -> "machine learning"
        """
        prompt = f"""
        You are a query refinement assistant.
        Your task is to extract the core search keywords from the user's natural language query for a book search API.

        RULES:
        - Remove conversational phrases like "recommend", "find books on", "show me", "what are good books about".
        - Keep only the CORE technical keywords or topic.
        - DO NOT ADD OR CHANGE LETTERS. NO TYPOS. Use exact spelling from the query.
        - Return ONLY the extracted keywords text.
        - Do NOT output any labels like "Keywords:" or "Result:".

        User Query: {query}
        Refined Keywords:"""

        try:
            refined = self.llm.invoke(prompt).content.strip()
            refined = refined.replace('"', '').replace("'", "")
            print(f"🔍 Refined Query: '{query}' -> '{refined}'")
            return refined
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"❌ Error refining book query: {e}")
            return query  # Fallback to original

    def process(self, query: str, **kwargs) -> str:
        """
        Search for books and return LLM-ranked recommendations.
        """
        # 1. Refine the query for better API results
        search_query = self._refine_query(query)

        # 2. Search both sources
        google_results = self.google_search.search(search_query, limit=5)
        openlibrary_results = self.openlibrary_search.search(search_query, limit=5)

        # 3. Merge and deduplicate
        merged_books = normalize_books(openlibrary_results, google_results)

        # 4. Fallback to original query if nothing found
        if not merged_books and search_query != query:
            print("⚠️ Refined search found nothing, retrying with original query...")
            google_results = self.google_search.search(query, limit=5)
            openlibrary_results = self.openlibrary_search.search(query, limit=5)
            merged_books = normalize_books(openlibrary_results, google_results)

        if not merged_books:
            return (
                f"⚠️ I couldn't find any books matching '{query}'. "
                "Try using more specific terms or a different topic."
            )

        # 5. LLM ranking and recommendation
        prompt = self.recommendation_prompt.format(
            query=query,
            books_data=json.dumps(merged_books, indent=2)
        )

        response = self.llm.invoke(prompt)
        return response.content


if __name__ == "__main__":
    load_dotenv()
    test_agent = BooksRecommendAgent()

    test_query = "Recommend books on deep learning"
    print(f"Query: {test_query}\n")

    test_result = test_agent.process(test_query)
    print(test_result)
