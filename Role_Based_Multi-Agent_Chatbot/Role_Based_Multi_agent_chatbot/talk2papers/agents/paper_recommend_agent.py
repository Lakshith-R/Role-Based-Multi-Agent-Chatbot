

import json
from dotenv import load_dotenv


from pathlib import Path
from agentic_student_assistant.core.base.base_agent import BaseAgent
from agentic_student_assistant.core.utils.config_loader import get_config
from agentic_student_assistant.core.utils.prompt_loader import load_agent_prompts
from agentic_student_assistant.talk2papers.tools.semantic_scholar_tool import SemanticScholarSearch
from agentic_student_assistant.talk2papers.tools.core_tool import CoreSearch
from agentic_student_assistant.talk2papers.tools.openreview_tool import OpenReviewSearch
from agentic_student_assistant.talk2papers.tools.arxiv_tool import ArXivSearch
from agentic_student_assistant.talk2papers.tools.paper_utils import normalize_papers
try:
    import agentlightning as agl
    AGL_AVAILABLE = True
except ImportError:
    AGL_AVAILABLE = False


class PaperRecommendAgent(BaseAgent):
    """
    Agent for recommending academic papers using Semantic Scholar and CORE.
    """
    
    def __init__(self):
        """Initialize paper recommendation agent."""
        config = get_config()
        super().__init__(config, agent_name="papers")
        # Load local prompts
        agent_path = Path(__file__).parent.parent
        prompts = load_agent_prompts(agent_path)
        self.recommendation_prompt = prompts['paper_recommendation_academic']
        self.qa_prompt = prompts['paper_qa_task']
        self.ss_search = SemanticScholarSearch()
        self.core_search = CoreSearch()
        self.openreview_search = OpenReviewSearch()
        self.arxiv_search = ArXivSearch()
    

    
    def _refine_query(self, query: str) -> str:
        """
        Extract core search terms from natural language query.
        Example: "tell me about biobridge paper" -> "biobridge"
        """
        prompt = f"""
        You are a query refinement assistant.
        Your task is to extract the core search keywords from the user's natural language query for an academic paper search API.
        
        RULES:
        - Remove conversational phrases like "tell me about", "find papers on", "show me", "what is".
        - Keep only the CORE technical keywords or title.
        - DO NOT ADD OR CHANGE LETTERS. NO TYPOS. Use exact spelling from the query.
        - If the user uses a specific term from a previous result, use that exact term.
        - Return ONLY the extracted keywords text.
        - Do NOT output any labels like "Keywords:" or "Result:".
        
        User Query: {query}
        Refined Keywords:"""
        
        try:
            refined = self.llm.invoke(prompt).content.strip()
            # Remove quotes if LLM added them
            refined = refined.replace('"', '').replace("'", "")
            print(f"🔍 Refined Query: '{query}' -> '{refined}'")
            return refined
        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"❌ Error in paper search: {e}")
            return query  # Fallback to original

    def process(self, query: str, **kwargs) -> str: # pylint: disable=too-many-branches
        """
        Summarize paper findings using LLM.query with refinement.
        """
        chat_history = kwargs.get("chat_history", [])
        
        # 1. CHECK CONTEXT: Is this a follow-up specific question? ("explain the first one")
        if self._is_selection_query(query):
            # A. Try history first
            if chat_history:
                # Basic check: does history actually contain paper info?
                last_bot_msg = next(
                    (msg for role, msg in reversed(chat_history) if role == "assistant"), ""
                )
                if len(last_bot_msg) > 100:  # Heuristic: if message is long enough
                    return self._explain_selection(query, chat_history)

            # B. If history empty or fails, treat as "Search & Explain"
            # e.g. "Explain the paper 'BioBridge'" -> Search BioBridge -> Explain result #1
            return self._search_and_explain(query)

        # 2. Refine query to get better API results
        search_query = self._refine_query(query)
        
        # 3. Search (Semantic Scholar + CORE)
        ss_results = self.ss_search.search(search_query, limit=3)
        core_results = self.core_search.search(search_query, limit=3)
        arxiv_results = self.arxiv_search.search(search_query, limit=3)
        
        # Track errors but don't return yet
        has_rate_limit = any(
            isinstance(res, dict) and res.get("error") == "rate_limit"
            for res in ss_results + core_results
        )
        has_forbidden = any(
            isinstance(res, dict) and res.get("error") == "forbidden"
            for res in ss_results + core_results
        )

        merged_papers = normalize_papers(ss_results, core_results, arxiv_results)
        
        # 4. Fallback search (OpenReview)
        if not merged_papers:
            print("🕒 Trying OpenReview as fallback...")
            or_results = self.openreview_search.search(search_query, limit=3)
            merged_papers = normalize_papers(merged_papers, or_results)
        
        # 4. Keyword Filtering (Strict Relevance Check)
        # Verify that either title or abstract contains some part of the refined query
        if search_query.lower() not in ["paper", "research", "study"]: # Don't filter if query is too generic
            keywords = search_query.lower().split()
            filtered_papers = []
            for p in merged_papers:
                text = ((p.get("title") or "") + " " + (p.get("abstract") or "")).lower()
                # Keep if at least one meaningful keyword is present
                if any(k in text for k in keywords if len(k) > 3):
                    filtered_papers.append(p)
                elif len(keywords) == 0:  # If no specific keywords, keep all
                    filtered_papers.append(p)
            
            if filtered_papers:
                merged_papers = filtered_papers

        # Pass ORIGINAL query to LLM for final ranking context
        if not merged_papers:
            # Try fallback to original query if refined one failed
            if search_query != query:
                print("⚠️ Refined search failed, trying original query...")
                ss_results = self.ss_search.search(query, limit=5)
                core_results = self.core_search.search(query, limit=5)
                merged_papers = normalize_papers(ss_results, core_results)

        if not merged_papers:
            if has_rate_limit:
                return (
                    "⚠️ **Rate Limit Reached**: The Semantic Scholar API is currently "
                    "limiting requests. I tried to find fallback results but found nothing. "
                    "Please wait a moment or try again later."
                )
            if has_forbidden:
                return (
                    "⚠️ **Access Forbidden**: Semantic Scholar has restricted access (403). "
                    "I tried to find fallback results but found nothing. Try a different query."
                )
            return f"⚠️ I couldn't find any academic papers matching '{query}'. Try broader terms."
        
        # 5. LLM Ranking (Use original query)
        prompt = self.recommendation_prompt.format(
            query=query,
            papers_data=json.dumps(merged_papers, indent=2)
        )
        
        # Agent Lightning Instrumentation
        if AGL_AVAILABLE:
            with agl.trace(name="paper_recommend_process", agent_name="paper_agent") as trace:
                agl.emit_prompt(prompt, tags=["papers", "recommendation"])
                response = self.llm.invoke(prompt)
                agl.emit_llm_response(response.content)
        else:
            response = self.llm.invoke(prompt)
            
        return response.content

    def _is_selection_query(self, query: str) -> bool:
        """Heuristic to check if user wants details on a previous result."""
        q = query.lower()
        triggers = [
            "explain", "tell me more", "elaborate", "summary of", "what about",
            "the first one", "second one", "#1", "#2", "this paper",
            "in this paper", "in the paper"
        ]
        return any(t in q for t in triggers)

    def _explain_selection(self, query: str, history: list) -> str:
        """Explain a paper from conversation history."""
        # Get last assistant message
        last_bot_msg = next((msg for role, msg in reversed(history) if role == "assistant"), "")
        
        prompt = f"""
        You are an academic expert. 
        The user is asking a follow-up question about a paper you just listed.
        
        USER QUERY: "{query}"
        
        YOUR PREVIOUS MESSAGE (Context):
        {last_bot_msg}
        
        TASK:
        Identify which paper the user is referring to (e.g., "first one", "BioBridge", "#2").
        Provide a detailed explanation of that paper based on the context or your internal knowledge.
        If you are unsure which paper, ask for clarification.
        """
        return self.llm.invoke(prompt).content

    def _search_and_explain(self, query: str) -> str:
        """
        Search for a specific paper and explain it (QA mode), instead of listing valid matches.
        Used when user asks "Explain paper X" but we have no history context.
        """
        # 1. Extract Paper Title/Keywords
        search_query = self._refine_query(query)
        print(f"🔍 [QA Mode] Searching for specific paper: {search_query}")

        # 2. Search
        ss_results = self.ss_search.search(search_query, limit=1) # Just get top match
        core_results = self.core_search.search(search_query, limit=1)
        arxiv_results = self.arxiv_search.search(search_query, limit=1)
        
        # Track errors but don't return yet
        has_rate_limit = any(
            isinstance(res, dict) and res.get("error") == "rate_limit"
            for res in ss_results + core_results
        )
        has_forbidden = any(
            isinstance(res, dict) and res.get("error") == "forbidden"
            for res in ss_results + core_results
        )
        for res in ss_results + core_results:
            if isinstance(res, dict) and res.get("error") == "timeout":
                print(f"⚠️ Search timeout encountered for: {search_query}")
        
        merged_papers = normalize_papers(ss_results, core_results, arxiv_results)
        
        # 3. Fallback to OpenReview if needed
        if not merged_papers:
            print(f"🕒 Search & Explain: Trying OpenReview fallback for '{search_query}'")
            or_results = self.openreview_search.search(search_query, limit=1)
            merged_papers = normalize_papers(merged_papers, or_results)
        
        if not merged_papers:
            if has_rate_limit:
                return (
                    "⚠️ **Rate Limit Reached**: The Semantic Scholar API is currently "
                    "limiting requests. I tried fallbacks but found no matches for this "
                    "specific paper."
                )
            if has_forbidden:
                return (
                    "⚠️ **Access Forbidden**: Semantic Scholar has restricted access (403). "
                    "I tried to find fallback results but found nothing for this specific paper."
                )
            return f"⚠️ I tried to find the paper '{search_query}' to explain it, but found no matches."
        
        target_paper = merged_papers[0]  # Take the best match
        
        # 3. Generate QA / Explanation
        prompt = self.qa_prompt.format(
            query=query,
            paper_data=json.dumps(target_paper, indent=2)
        )
        
        return self.llm.invoke(prompt).content

if __name__ == "__main__":
    load_dotenv()
    test_agent = PaperRecommendAgent()
    
    test_query = "Find recent papers on Large Language Models"
    print(f"Query: {test_query}\n")
    
    test_result = test_agent.process(test_query)
    print(test_result)
