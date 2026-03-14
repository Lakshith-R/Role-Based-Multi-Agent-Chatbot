"""
Base agent class for all specialist agents.
Provides common interface and shared functionality.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from omegaconf import DictConfig
from langchain_community.chat_models import ChatOpenAI
from agentic_student_assistant.core.utils.llm_factory import LLMFactory

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    
    All specialist agents should inherit from this class and implement
    the process() method.
    """
    
    def __init__(self, agent_config: DictConfig, agent_name: str = "base", llm: Any = None):
        """
        Initialize base agent.
        
        Args:
            agent_config: Application configuration from Hydra
            agent_name: Name of the agent (for logging/debugging)
            llm: Optional LLM instance for dependency injection (testing)
        """
        self.config = agent_config
        self.agent_name = agent_name
        self.llm = llm if llm is not None else self._init_llm()
    
    def _init_llm(self) -> ChatOpenAI:
        """
        Initialize LLM from configuration.
        
        Returns:
            ChatOpenAI instance
        """
        return LLMFactory.create_llm(self.config.models)
    
    @abstractmethod
    def process(self, query: str, **kwargs) -> str:
        """
        Process a query and return a response.
        
        This method must be implemented by all subclasses.
        
        Args:
            query: User query
            **kwargs: Additional parameters specific to the agent
            
        Returns:
            str: Agent response
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get agent metadata (useful for debugging/logging).
        
        Returns:
            Dict with agent information
        """
        return {
            'agent_name': self.agent_name,
            'model': self.llm.model_name,
            'temperature': self.llm.temperature,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.agent_name}')"


if __name__ == "__main__":
    from agentic_student_assistant.core.utils.config_loader import get_config
    
    # Test with a simple implementation
    class TestAgent(BaseAgent):
        def process(self, query: str, **kwargs) -> str:
            return f"Processed: {query}"
    
    config = get_config()
    agent = TestAgent(config, agent_name="test")
    
    print(f"🤖 Created: {agent}")
    print(f"📋 Metadata: {agent.get_metadata()}")
    print(f"✅ Result: {agent.process('Hello world')}")
