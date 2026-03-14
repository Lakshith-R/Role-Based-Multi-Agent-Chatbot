"""
LLM Factory for creating LLM instances from configuration.
Supports OpenAI (paid) and Groq (free) with automatic fallback.
"""
import os
from typing import Optional
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI # pylint: disable=no-name-in-module,import-error
from langchain.embeddings import OpenAIEmbeddings
from omegaconf import DictConfig


class LLMFactory:
    """
    Factory for creating LLM and embedding instances.
    Automatically falls back to Groq if OpenAI API key is not available.
    """
    
    @classmethod
    def _get_provider(cls) -> str:
        """
        Determine which LLM provider to use.
        
        Returns:
            str: 'openai' or 'groq'
        """
        # User requested Groq-only mode
        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if groq_key:
            print("✅ Using Groq (Free Tier) as requested")
            return "groq"
        
        if openai_key and not openai_key.startswith("sk-your-"):
            print("✅ Groq key missing. Using OpenAI (paid)")
            return "openai"
        
        raise ValueError(
                "❌ No valid API key found! Please set:\n"
                "  - GROQ_API_KEY (Required for this configuration)\n"
                "Get Groq key: https://console.groq.com/keys"
            )
    
    @classmethod
    def create_llm(
        cls,
        model_config: DictConfig,
        temperature: Optional[float] = None,
        streaming: Optional[bool] = None
    ) -> ChatOpenAI:
        """
        Create LLM instance with automatic provider detection.
        
        Args:
            model_config: Model configuration from Hydra
            temperature: Override temperature (optional)
            streaming: Override streaming setting (optional)
            
        Returns:
            ChatOpenAI or compatible LLM instance
        """
        provider = cls._get_provider()
        
        if provider == "openai":
            return cls._create_openai_llm(model_config, temperature, streaming)
        
        if provider == "groq":
            return cls._create_groq_llm(model_config, temperature, streaming)
            
        return None
    
    @classmethod
    def _create_openai_llm(
        cls,
        model_config: DictConfig,
        temperature: Optional[float],
        streaming: Optional[bool]
    ) -> ChatOpenAI:
        """Create OpenAI LLM instance."""
        return ChatOpenAI(
            model=model_config.name,
            temperature=temperature if temperature is not None else model_config.temperature,
            max_tokens=model_config.max_tokens,
            streaming=streaming if streaming is not None else model_config.streaming
        )
    
    @classmethod
    def _create_groq_llm(cls, model_config: DictConfig, temperature: Optional[float], streaming: Optional[bool]):
        """
        Create Groq LLM instance (compatible with LangChain).
        Uses ChatOpenAI with custom base_url for Groq API.
        """
        # Map GPT models to Groq equivalents
        model_mapping = {
            "gpt-4-turbo": "llama-3.3-70b-versatile",
            "gpt-3.5-turbo": "llama-3.1-8b-instant",
        }
        
        requested_model = model_config.name
        groq_model = model_mapping.get(requested_model, "llama-3.3-70b-versatile")
        
        print(f"💡 Mapping {requested_model} → Groq {groq_model}")
        
        return ChatOpenAI(
            model=groq_model,
            temperature=temperature if temperature is not None else model_config.temperature,
            max_tokens=model_config.max_tokens,
            openai_api_key=os.getenv("GROQ_API_KEY"),
            openai_api_base="https://api.groq.com/openai/v1",
            streaming=streaming if streaming is not None else False
        )
    
    @staticmethod
    def create_chat_model(
        model_name: str = "gpt-4",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        streaming: bool = False
    ) -> ChatOpenAI:
        """
        Quick helper to create a chat model with explicit parameters.
        
        Args:
            model_name: Model name
            temperature: Temperature setting
            max_tokens: Max tokens
            streaming: Enable streaming
            
        Returns:
            ChatOpenAI: Configured LLM instance
        """
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming
        )
    
    @staticmethod
    def create_embeddings(model_config: DictConfig):
        """
        Create embeddings instance.
        Falls back to HuggingFace (local/free) if OpenAI key is invalid.
        """
        openai_key = os.getenv("OPENAI_API_KEY")
        
        # Check if OpenAI key is valid (not empty and not the default placeholder)
        is_openai_valid = openai_key and not openai_key.startswith("sk-your-")
        
        if is_openai_valid:
            embeddings_model = model_config.get(
                "embedding_model", "text-embedding-3-small"
            )
            return OpenAIEmbeddings(model=embeddings_model)
            
        print("⚠️ Using HuggingFace Embeddings (Free/Local) because OpenAI key is missing.")
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings # pylint: disable=import-outside-toplevel
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except ImportError as exc:
            raise ImportError(
                "❌ 'sentence-transformers' not found! Please run: pip install sentence-transformers"
            ) from exc


if __name__ == "__main__":
    from agentic_student_assistant.core.utils.config_loader import get_config
    
    # Test LLM creation
    config = get_config()
    
    print("🤖 Creating LLM from config...")
    llm = LLMFactory.create_llm(config.models)
    print(f"✅ Created: {llm.model_name} (temp={llm.temperature})")
    
    print("\n🔢 Creating embeddings...")
    try:
        embeddings = LLMFactory.create_embeddings(config.models)
        print(f"✅ Created: {embeddings.model}")
    except Exception as e: # pylint: disable=broad-exception-caught
        print(f"⚠️ Embeddings error: {e}")
    
    # Test with override
    print("\n🤖 Creating LLM with temperature override...")
    llm_hot = LLMFactory.create_llm(config.models, temperature=0.7)
    print(f"✅ Created: {llm_hot.model_name} (temp={llm_hot.temperature})")
    
    # Test actual query
    print("\n🧪 Testing LLM with a query...")
    try:
        response = llm.invoke("Say 'Hello from LLM factory!'")
        print(f"✅ Response: {response.content}")
    except Exception as e: # pylint: disable=broad-exception-caught
        print(f"❌ Error: {e}")
