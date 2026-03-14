"""
Utility to load agent-specific prompts from local config files.
"""
import yaml
from pathlib import Path
from typing import Dict


def load_agent_prompts(agent_path: Path) -> Dict[str, str]:
    """
    Load prompts from an agent's local configs/prompts.yaml file.
    
    Args:
        agent_path: Path to the agent directory (e.g., agentic_student_assistant/talk2jobs)
        
    Returns:
        Dictionary of prompt templates
    """
    prompts_file = agent_path / "configs" / "prompts.yaml"
    
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    
    with open(prompts_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
