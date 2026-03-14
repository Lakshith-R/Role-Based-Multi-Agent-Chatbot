"""
Job market agent for searching jobs.
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st


from agentic_student_assistant.core.base.base_agent import BaseAgent
from agentic_student_assistant.core.utils.config_loader import get_config
from agentic_student_assistant.core.utils.prompt_loader import load_agent_prompts
from agentic_student_assistant.talk2jobs.tools.google_search import GoogleSearch
try:
    import agentlightning as agl
    AGL_AVAILABLE = True
except ImportError:
    AGL_AVAILABLE = False

load_dotenv()


class JobMarketAgent(BaseAgent):
    """
    Agent for searching and analyzing job market data.
    """
    
    def __init__(self):
        """Initialize job market agent."""
        config = get_config()
        super().__init__(config, agent_name="job_market")
        # Load local prompts
        agent_path = Path(__file__).parent.parent
        prompts = load_agent_prompts(agent_path)
        self.analysis_prompt = prompts['job_market_analysis']
    
    def _extract_location(self, query: str) -> tuple:
        """
        Extract location from query if present.
        Returns (cleaned_query, location) tuple.
        """
        # List of fields to avoid capturing as location
        field_keywords = [
            "data science", "machine learning", "ai", "artificial intelligence",
            "software engineer", "web development", "frontend", "backend",
            "devops", "cloud", "cybersecurity", "data analyst", "data engineer",
            "mobile development", "ios", "android", "fullstack", "full stack",
            "python", "java", "javascript", "react", "node", "django"
        ]
        
        # Common location markers (lowercase for easier matching)
        query_lower = query.lower()
        markers = [" in ", " at ", " near "]
        
        for marker in markers:
            if marker in query_lower:
                # Find the actual case marker in the original query
                idx = query_lower.rfind(marker)
                potential_loc = query[idx + len(marker):].strip()
                
                # Check if it's not just a technical field
                if potential_loc and potential_loc.lower() not in field_keywords:
                    cleaned_query = query[:idx].strip()
                    return cleaned_query, potential_loc
        
        return query, None
    
    def search_jobs(self, query: str, field: str = None) -> list:
        """
        Search for job listings using SerpAPI Google Jobs API.
        
        Args:
            query: Full user query
            field: Extracted field/domain (e.g., "data science")
            
        Returns:
            List of job listings with structured data
        """
        # Extract location if mentioned in query
        cleaned_query, location = self._extract_location(query)
        
        # Build API parameters
        search_q = field if field else cleaned_query
        if search_q and "jobs" not in search_q.lower() and "career" not in search_q.lower():
            search_q = f"{search_q} jobs"
            
        params = {
            "engine": "google_jobs",
            "q": search_q if search_q else "jobs",
            "api_key": os.getenv("SERPAPI_API_KEY") or st.secrets.get("SERPAPI_API_KEY"),
        }
        
        # Add location and regional parameters if found
        if location:
            # Capitalize first letter of each word for better API match
            location = location.title()
            params["location"] = location
            
            # Regional mapping for better international results
            region_map = {
                "germany": {"gl": "de", "hl": "de", "google_domain": "google.de"},
                "berlin": {"gl": "de", "hl": "de", "google_domain": "google.de"},
                "munich": {"gl": "de", "hl": "de", "google_domain": "google.de"},
                "japan": {"gl": "jp", "hl": "ja", "google_domain": "google.co.jp"},
                "tokyo": {"gl": "jp", "hl": "ja", "google_domain": "google.co.jp"},
                "india": {"gl": "in", "hl": "en", "google_domain": "google.co.in"},
                "bangalore": {"gl": "in", "hl": "en", "google_domain": "google.co.in"},
                "uk": {"gl": "uk", "hl": "en", "google_domain": "google.co.uk"},
                "london": {"gl": "uk", "hl": "en", "google_domain": "google.co.uk"},
                "usa": {"gl": "us", "hl": "en", "google_domain": "google.com"},
                "mexico": {"gl": "mx", "hl": "es", "google_domain": "google.com.mx"},
                "canada": {"gl": "ca", "hl": "en", "google_domain": "google.ca"},
                "france": {"gl": "fr", "hl": "fr", "google_domain": "google.fr"},
                "spain": {"gl": "es", "hl": "es", "google_domain": "google.es"},
                "brazil": {"gl": "br", "hl": "pt", "google_domain": "google.com.br"},
                "australia": {"gl": "au", "hl": "en", "google_domain": "google.com.au"},
            }
            
            loc_lower = location.lower()
            for key, region_params in region_map.items():
                if key in loc_lower:
                    params.update(region_params)
                    break
        
        search = GoogleSearch(params)
        try:
            results = search.get_dict()
        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"❌ SerpAPI Error: {e}")
            if "400" in str(e):
                return [{
                    "error": "invalid_request",
                    "message": "The search request was invalid. Try refining your query."
                }]
            return []
        
        # Check if error in results
        if "error" in results:
            print(f"❌ SerpAPI Result Error: {results['error']}")
            return []
        
        # Extract actual job postings
        jobs = results.get("jobs_results", [])
        
        # Fallback: If no results with regional params, try a global search without hl/gl/domain
        if not jobs and ("gl" in params or "google_domain" in params):
            print("🔄 No results found with regional params. Falling back to global search...")
            fallback_params = {
                "engine": "google_jobs",
                "q": f"{search_q} in {location}" if location else search_q,
                "api_key": params["api_key"],
                "hl": "en"
            }
            try:
                fallback_search = GoogleSearch(fallback_params)
                fallback_results = fallback_search.get_dict()
                jobs = fallback_results.get("jobs_results", [])
            except Exception as e: # pylint: disable=broad-exception-caught
                print(f"❌ Fallback Error: {e}")
        if not jobs:
            return []
        
        # Extract structured job data
        job_listings = []
        for job in jobs[:3]:  # Get top 3 jobs to save tokens
            # Get primary apply link
            apply_options = job.get("apply_options", [])
            apply_link = apply_options[0].get("link") if apply_options else job.get("share_link", "No link")
            
            job_listings.append({
                "title": job.get("title", "No title"),
                "company": job.get("company_name", "Unknown company"),
                "location": job.get("location", "Location not specified"),
                "link": apply_link,
                "snippet": job.get("description", "")[:200] + "..." if job.get("description") else "",
                "via": job.get("via", ""),
                # Additional useful fields
                "salary": job.get("detected_extensions", {}).get("salary"),
                "job_type": job.get("detected_extensions", {}).get("schedule_type"),
                "posted_at": job.get("detected_extensions", {}).get("posted_at")
            })
        
        return job_listings
    
    def summarize_jobs(self, job_listings: list) -> str:
        """
        Summarize job listings using GPT.
        
        Args:
            job_listings: List of job data
            
        Returns:
            Summary of jobs
        """
        if not job_listings:
            return "⚠️ No job data to summarize."
        
        prompt = f"{self.analysis_prompt}\n\nHere are the listings:\n{json.dumps(job_listings, indent=2)}"
        
        # Agent Lightning Instrumentation
        if AGL_AVAILABLE:
            with agl.trace(name="job_market_summarize", agent_name="job_agent") as trace:
                agl.emit_prompt(prompt, tags=["jobs", "analysis"])
                response = self.llm.invoke(prompt)
                agl.emit_llm_response(response.content)
        else:
            response = self.llm.invoke(prompt)

        return response.content
    
    def _extract_field_from_query(self, query: str) -> str:
        """
        Extract the field/domain from user query.
        Returns the field name or empty string if not found.
        """
        query_lower = query.lower()
        
        # Common field keywords
        fields = [
            "data science", "machine learning", "ai", "artificial intelligence",
            "software engineer", "web development", "frontend", "backend",
            "devops", "cloud", "cybersecurity", "data analyst", "data engineer",
            "mobile development", "ios", "android", "fullstack", "full stack",
            "python", "java", "javascript", "react", "node", "django"
        ]
        
        for field in fields:
            # Use a slightly more flexible check for field in query
            if field in query_lower:
                return field
        
        # Check if query has job-related keywords without specific field
        job_keywords = ["job", "career", "position", "opportunity", "opportunities", "hiring"]
        has_job_keyword = any(keyword in query_lower for keyword in job_keywords)
        
        if has_job_keyword:
            return ""  # Job query but no specific field
        
        return query  # Assume the whole query is the field
    
    def process(self, query: str, **kwargs) -> str:
        """
        Process job market query.
        
        Args:
            query: Job search query
            **kwargs: Additional parameters
            
        Returns:
            Formatted job analysis with listings
        """
        # Check if user specified a field/domain
        field = self._extract_field_from_query(query)
        
        # Extract location for display purposes
        _, location = self._extract_location(query)
        
        # If no specific field mentioned, ask for clarification
        if field == "":
            return """### Which field are you interested in?
                    Please specify the domain or technology you'd like to see job listings for. For example:
                    - "Data Science jobs"
                    - "Software Engineering positions"
                    - "Machine Learning opportunities"
                    - "Web Development careers"
                    - "Cybersecurity jobs"
                    - "Mobile App Development"

                    **Example queries:**
                    - "Show me data science jobs"
                    - "Find frontend developer positions"
                    - "Career opportunities in AI"

                    💡 **Tip**: The more specific you are, the better the job matches!"""
        
        # Search for jobs with the field
        listings = self.search_jobs(query, field=field)
        if not listings:
            # location is already extracted above
            location_msg = f" in **{location}**" if location else ""
            return f"❌ No job listings found for **{field}**{location_msg}. Try:\n" \
                   f"- A different location\n" \
                   f"- Broader search terms\n" \
                   f"- Check spelling of location/job title"
        
        # Check if listings is an error response
        if isinstance(listings, list) and len(listings) > 0 and "error" in listings[0]:
            return f"⚠️ **Search Error**: {listings[0].get('message', 'Unknown error occurred.')}\n\n" \
                   f"Try making your query more specific (e.g., 'Data Science jobs in Berlin')."
        
        # Save fresh listings to file
        os.makedirs("data", exist_ok=True)
        with open("data/job_listings.json", "w", encoding="utf-8") as f:
            json.dump(listings, f, indent=2, ensure_ascii=False)
        
        # Build display output using LLM (for translation and formatting)
        display_output = self.summarize_jobs(listings)
        
        header = f"{field.title()} Job Opportunities in {location if location else 'Global'}\n\n"
        return header + display_output


# Legacy function for backward compatibility
def run_job_market_agent(query: str) -> str:
    """Legacy entry point."""
    agent = JobMarketAgent()
    return agent.process(query)


if __name__ == "__main__":
    test_agent = JobMarketAgent()
    
    test_query = input("\n🔎 Enter job search query: ")
    test_result = test_agent.process(test_query)
    
    print("Job Market Analysis:")
    print(test_result)
