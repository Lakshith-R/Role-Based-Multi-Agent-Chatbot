"""
Fallback agent for handling general queries outside specialist domains.
Refactored to inherit from BaseAgent and use config-based architecture.
"""
from dotenv import load_dotenv


from agentic_student_assistant.core.base.base_agent import BaseAgent
from agentic_student_assistant.core.utils.config_loader import get_config, get_prompt

load_dotenv()


class FallbackAgent(BaseAgent):
    """
    Fallback agent for general queries that don't fit specialist agents.
    Uses GPT with general knowledge to provide helpful responses.
    """
    
    def __init__(self):
        """Initialize fallback agent with configuration."""
        config = get_config()
        super().__init__(config, agent_name="fallback")
        self.system_prompt = get_prompt("fallback_general")
    
    def process(self, query: str, **kwargs) -> str:
        """
        Process a general query using GPT.
        
        Args:
            query: User query
            **kwargs: Additional parameters (unused)
            
        Returns:
str: Agent response
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query}
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def run(self, query: str) -> str:
        """Legacy method for backward compatibility."""
        return self.process(query)


if __name__ == "__main__":
    agent = FallbackAgent()
    
    test_query = "What is the capital of France?"
    result = agent.process(test_query)
    
    print("\n🤖 FallbackAgent Test")
    print(f"Query: {test_query}")
    print(f"Response: {result}")
