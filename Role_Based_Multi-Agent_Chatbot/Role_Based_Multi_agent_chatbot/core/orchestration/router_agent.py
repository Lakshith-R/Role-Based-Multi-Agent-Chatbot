"""
LLM-based router agent with structured output.
Replaces keyword-based routing with semantic understanding.
"""
from typing import Literal
# pylint: disable=no-name-in-module
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser


from agentic_student_assistant.core.utils.config_loader import get_config, get_prompt
from agentic_student_assistant.core.utils.llm_factory import LLMFactory


class RouteDecision(BaseModel):
    """Structured output for routing decisions."""
    
    agent: Literal["job_market", "books", "papers", "documents", "orchestrator", "fallback"] = Field(
        description="The agent that should handle this query"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )
    reasoning: str = Field(
        description="Brief explanation of why this agent was selected"
    )


class RouterAgent:
    """
    LLM-based router that uses GPT to intelligently classify queries.
    Provides semantic understanding, confidence scoring, and reasoning.
    """
    
    def __init__(self):
        """Initialize router with configuration."""
        self.config = get_config()
        self.llm = self._init_llm()
        self.parser = PydanticOutputParser(pydantic_object=RouteDecision)
        self.prompt = self._create_prompt()
        self.chain = self.prompt | self.llm | self.parser # pylint: disable=unsupported-binary-operation
    
    def _init_llm(self) -> ChatOpenAI:
        """Initialize LLM for routing."""
        # Use higher temperature for routing to allow flexibility
        return LLMFactory.create_llm(
            self.config.models,
            temperature=0.1  # Low temperature for consistent routing
        )
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create routing prompt template."""
        system_prompt = get_prompt("router_system")
        format_instructions = self.parser.get_format_instructions()
        
        # Use a template with placeholders, then partial them out
        # This prevents braces in the JSON schema/system prompt from being interpreted as variables
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}\n\nRecent Conversation:\n{chat_history}\n\n{format_instructions}"),
            ("user", "{query}")
        ])
        
        return prompt.partial(
            system_prompt=system_prompt,
            format_instructions=format_instructions,
            chat_history="" # Default empty if not provided
        )
    
    def route(self, query: str, chat_history: str = "") -> RouteDecision:
        """
        Route a query to the appropriate agent.
        
        Args:
            query: User query to route
            chat_history: Previous conversation context
            
        Returns:
            RouteDecision with agent, confidence, and reasoning
        """
        try:
            # Format history string if list provided (simple heuristic)
            history_str = str(chat_history)[-1000:] if chat_history else "No history."
            
            decision = self.chain.invoke({
                "query": query,
                "chat_history": history_str
            })
            
            # Apply confidence threshold from config
            threshold = self.config.routing.confidence_threshold
            if decision.confidence < threshold and self.config.routing.fallback_on_low_confidence:
                print(f"⚠️ Low confidence ({decision.confidence:.2f}), routing to fallback")
                decision.agent = "fallback"
                decision.reasoning = f"Low confidence ({decision.confidence:.2f}). " + decision.reasoning
            
            return decision
        
        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"❌ Routing error: {e}")
            # Fallback to safe default
            return RouteDecision(
                agent="fallback",
                confidence=0.0,
                reasoning=f"Error during routing: {str(e)}"
            )
    
    def route_with_orchestration(self, query: str, chat_history: str = "") -> RouteDecision:
        """
        Route with orchestration detection for complex queries.
        
        Detects if a query requires multiple agents and routes to orchestrator.
        
        Args:
            query: User query
            chat_history: Previous conversation context
            
        Returns:
            RouteDecision
        """
        decision = self.route(query, chat_history)
        
        # Check if query mentions multiple domains (heuristic for orchestration)
        query_lower = query.lower()
        domains = 0
        if any(word in query_lower for word in ["job", "career", "hiring", "work"]):
            domains += 1
        if any(word in query_lower for word in ["paper", "article", "citation", "research", "study"]):
            domains += 1
        if any(word in query_lower for word in ["book", "reading", "textbook", "resource", "bibtex"]):
            domains += 1
        
        # If multiple domains detected, consider orchestrator
        if domains >= 2:
            print(f"🎭 Multiple domains detected ({domains}), routing to orchestrator")
            decision.agent = "orchestrator"
            decision.reasoning = f"Complex query spanning {domains} domains ({query}). " + decision.reasoning
        
        return decision


# Singleton instance
_router: 'RouterAgent' = None


def get_router() -> 'RouterAgent':
    """Get global router instance (singleton)."""
    global _router # pylint: disable=global-statement
    if _router is None:
        _router = RouterAgent()
    return _router


def route_query(query: str, enable_orchestration: bool = False, chat_history: str = "") -> RouteDecision:
    """
    Route a query to the appropriate agent.
    
    Args:
        query: User query
        enable_orchestration: Whether to detect and route complex queries to orchestrator
        chat_history: Previous conversation context
        
    Returns:
        RouteDecision with agent, confidence, and reasoning
    """
    router = get_router()
    
    if enable_orchestration:
        return router.route_with_orchestration(query, chat_history)

    return router.route(query, chat_history)


if __name__ == "__main__":
    print("🧪 Testing Router Agent\n")
    
    # Test queries
    test_queries = [
        "Find AI jobs in Berlin",
        "Recommend books on Python programming",
        "What's the weather today?",
        "Find papers and books to learn about Transformers for my next job",  # Multi-domain
    ]
    
    for test_query in test_queries:
        print(f"\n❓ Query: {test_query}")
        test_decision = route_query(test_query, enable_orchestration=True)
        print(f"   🎯 Agent: {test_decision.agent}")
        print(f"   📊 Confidence: {test_decision.confidence:.2f}")
        print(f"   💭 Reasoning: {test_decision.reasoning}")
        print("-" * 70)
