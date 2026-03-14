"""
Configuration loader using Hydra for centralized config management.
Provides singleton access to configuration throughout the application.
"""
from omegaconf import DictConfig
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from pathlib import Path
from typing import Optional


class ConfigManager:
    """Singleton configuration manager using Hydra."""
    
    _config: Optional[DictConfig] = None
    _initialized = False
    
    @classmethod
    def get_config(cls) -> DictConfig:
        """
        Get the application configuration.
        Initializes Hydra on first call and caches the config.
        
        Returns:
            DictConfig: The application configuration
        """
        if cls._config is None:
            cls._initialize()
        return cls._config
    
    @classmethod
    def _initialize(cls):
        """Initialize Hydra configuration."""
        if cls._initialized:
            return
            
        # Get absolute path to config directory
        config_dir = Path(__file__).parent.parent / "configs"
        config_dir = config_dir.resolve()
        
        if not config_dir.exists():
            raise RuntimeError(f"Config directory not found: {config_dir}")
        
        try:
            # Clear any existing Hydra instance to avoid "already initialized" errors
            GlobalHydra.instance().clear()
            
            # Initialize Hydra with absolute path
            initialize_config_dir(
                version_base=None,
                config_dir=str(config_dir)
            )
            cls._config = compose(config_name="config")
            cls._initialized = True
            print(f"✅ Configuration loaded from: {config_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}") from e
    
    @classmethod
    def get_prompt(cls, prompt_name: str) -> str:
        """
        Get a prompt template by name.
        
        Args:
            prompt_name: Name of the prompt (e.g., 'curriculum_qa', 'router_system')
            
        Returns:
            str: The prompt template
        """
        app_config = cls.get_config()
        prompt = app_config.prompts.get(prompt_name, "")
        if not prompt:
            print(f"⚠️ Warning: Prompt '{prompt_name}' not found in config")
        return prompt
    
    @classmethod
    def reload(cls):
        """Reload configuration (useful for testing or config updates)."""
        cls._config = None
        cls._initialized = False
        cls.get_config()


# Convenience function for easy imports
def get_config() -> DictConfig:
    """Get application configuration."""
    return ConfigManager.get_config()


def get_prompt(prompt_name: str) -> str:
    """Get a prompt template by name."""
    return ConfigManager.get_prompt(prompt_name)


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print(f"\n📋 App Name: {config.app.name}")
    print(f"📋 App Version: {config.app.version}")
    print(f"📋 LLM Router Enabled: {config.routing.use_llm_router}")
    print(f"📋 Cache Enabled: {config.caching.enabled}")
    
    # Test prompt loading
    print("\n📝 Router System Prompt:")
    print(get_prompt("router_system")[:200] + "...")
