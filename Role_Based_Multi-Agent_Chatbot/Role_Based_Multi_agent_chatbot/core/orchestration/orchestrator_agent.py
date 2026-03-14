import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

from agentic_student_assistant.core.base.base_agent import BaseAgent
from agentic_student_assistant.core.utils.config_loader import get_config, get_prompt
from agentic_student_assistant.core.orchestration.router_agent import route_query

# We'll use the individual agents to fulfill sub-queries
from agentic_student_assistant.talk2jobs.agents.job_market_agent import run_job_market_agent
from agentic_student_assistant.talk2books.agents.books_recommend_agent import BooksRecommendAgent
from agentic_student_assistant.talk2papers.agents.paper_recommend_agent import PaperRecommendAgent
from agentic_student_assistant.talk2docs.agents.docs_agent import DocsRecommendAgent
from agentic_student_assistant.core.base.fallback_agent import FallbackAgent


class SubQuery(BaseModel):
    """A sub-query decomposed from a complex user request."""
    query: str = Field(description="The specific sub-question to answer")
    domain: str = Field(description="The domain of the query (e.g., 'jobs', 'books', 'papers', 'documents')")


class DecomposedPlan(BaseModel):
    """The full decomposition plan."""
    sub_queries: List[SubQuery] = Field(description="List of sub-queries to execute")


class OrchestratorAgent(BaseAgent):
    """
    Agent that handles complex, multi-domain queries by:
    1. Decomposing the request into sub-queries.
    2. Routing each sub-query to the appropriate specialized agent.
    3. Aggregating the results into a cohesive, final response.
    """

    def __init__(self):
        config = get_config()
        super().__init__(config, agent_name="orchestrator")
        
        self.decomposition_parser = PydanticOutputParser(pydantic_object=DecomposedPlan)
        
        # Instantiate agents for reuse
        self.books_agent = BooksRecommendAgent()
        self.papers_agent = PaperRecommendAgent()
        self.docs_agent = DocsRecommendAgent()
        self.fallback = FallbackAgent()

    def _decompose_query(self, query: str) -> DecomposedPlan:
        """Break down a complex query into specialized sub-queries."""
        system_prompt = get_prompt("orchestrator_system") + "\n\n{format_instructions}"
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{query}")
        ]).partial(format_instructions=self.decomposition_parser.get_format_instructions())

        chain = prompt | self.llm | self.decomposition_parser
        
        try:
            return chain.invoke({"query": query})
        except Exception as e:
            print(f"❌ Orchestrator decomposition failed: {e}")
            # Fallback to simple sub-queries if parsing fails
            return DecomposedPlan(sub_queries=[
                SubQuery(query=query, domain="fallback")
            ])

    def _execute_sub_queries(self, plan: DecomposedPlan, chat_history: List = None) -> List[Dict[str, str]]:
        """Run each sub-query through the appropriate agent and collect results."""
        results = []
        
        for sub in plan.sub_queries:
            # We use the router to verify the correct agent for the sub-query!
            decision = route_query(sub.query, enable_orchestration=False, chat_history=str(chat_history) if chat_history else "")
            target_agent = decision.agent
            
            print(f"🎵 Orchestrator executing sub-query: '{sub.query}' -> {target_agent}")
            
            try:
                if target_agent == "job_market":
                    result = run_job_market_agent(sub.query)
                elif target_agent == "books":
                    result = self.books_agent.process(sub.query)
                elif target_agent == "papers":
                    result = self.papers_agent.process(sub.query, chat_history=chat_history)
                elif target_agent == "documents":
                    result = self.docs_agent.process(sub.query, chat_history=chat_history)
                else:
                    result = self.fallback.run(sub.query)
            except Exception as e:
                result = f"Failed to get answer for this part: {e}"
                
            results.append({
                "query": sub.query,
                "agent": target_agent,
                "result": result
            })
            
        return results

    def _aggregate_results(self, original_query: str, agent_results: List[Dict[str, str]]) -> str:
        """Synthesize a final, cohesive answer from the individual agent results."""
        system_prompt = get_prompt("orchestrator_synthesis")
        if not system_prompt or system_prompt == "orchestrator_synthesis":
            # Fallback inline if it's not present in yaml yet
            system_prompt = (
                "You are the Lead Assistant coordinating responses from multiple expert agents.\n"
                "Synthesize their individual findings into a single, cohesive, and easy-to-read response that directly answers the user's original query.\n"
                "Use clear headings, bullet points, and a unified voice. Do not just blindly paste their responses; blend them naturally.\n"
                "If one agent failed to find something, focus on the successful findings.\n"
                "Do not hallucinate external information beyond what the agents provided."
            )
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "ORIGINAL QUERY: {query}\n\nEXPERT FINDINGS:\n{findings}")
        ])

        # Format findings for prompt
        findings_text = ""
        for i, res in enumerate(agent_results, 1):
            findings_text += f"--- EXPERT {i} ({res['agent'].upper()}) ---\n"
            findings_text += f"{res['result']}\n\n"

        chain = prompt | self.llm
        
        try:
            response = chain.invoke({
                "query": original_query,
                "findings": findings_text
            })
            return response.content
        except Exception as e:
            print(f"❌ Orchestrator aggregation failed: {e}")
            return "Gathered results but failed to synthesize them:\n\n" + findings_text

    def process(self, query: str, **kwargs) -> str:
        """Main entry point for OrchestratorAgent."""
        chat_history = kwargs.get("chat_history", [])
        
        # 1. Decompose
        print("\n🎼 Orchestrator is decomposing the complex query...")
        plan = self._decompose_query(query)
        
        # 2. Execute
        print("🎼 Orchestrator is farming out sub-queries to experts...")
        agent_results = self._execute_sub_queries(plan, chat_history)
        
        # 3. Aggregate
        print("🎼 Orchestrator is synthesizing final response...")
        final_answer = self._aggregate_results(query, agent_results)
        
        return final_answer

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    agent = OrchestratorAgent()
    test_query = "Find AI jobs in Berlin, and also recommend some textbooks to learn deep learning"
    print(f"Test Query: {test_query}\n")
    print(agent.process(test_query))
