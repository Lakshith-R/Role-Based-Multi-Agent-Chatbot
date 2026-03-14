"""
Unified logging manager supporting multiple handlers.
Replaces duplicate logging code across the application.
"""
import os
import datetime
from typing import Optional, List
from abc import ABC, abstractmethod


class LogHandler(ABC):
    """Abstract base class for log handlers."""
    
    @abstractmethod
    def log(self, **kwargs):
        """Log an interaction."""
        pass


class FileLogHandler(LogHandler):
    """Handler for logging to local files."""
    
    def __init__(self, log_dir: str = "logs", log_file: str = "workflow_logs.txt"):
        self.log_dir = log_dir
        self.log_file = log_file
        os.makedirs(log_dir, exist_ok=True)
    
    def log(self, **kwargs):
        """Write log entry to file."""
        log_path = os.path.join(self.log_dir, self.log_file)
        
        timestamp = kwargs.get('timestamp', datetime.datetime.now().isoformat())
        query = kwargs.get('query', '')
        agent = kwargs.get('agent', 'unknown')
        curriculum_mode = kwargs.get('curriculum_mode', 'srh')
        latency = kwargs.get('latency')
        is_fallback = kwargs.get('is_fallback', False)
        result = kwargs.get('result', '')
        confidence = kwargs.get('confidence')
        reasoning = kwargs.get('reasoning', '')
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"🕒 Timestamp: {timestamp}\n")
            f.write(f"❓ Query: {query}\n")
            f.write(f"📂 Curriculum Mode: {curriculum_mode}\n")
            f.write(f"📌 Routed Agent: {agent}\n")
            
            if confidence is not None:
                f.write(f"🎯 Confidence: {confidence:.2f}\n")
            if reasoning:
                f.write(f"💭 Reasoning: {reasoning}\n")
            if latency is not None:
                f.write(f"⏱️ Latency: {latency:.2f} seconds\n")
            
            f.write(f"🛡️ Fallback Used: {'Yes' if is_fallback else 'No'}\n")
            f.write("📘 Final Answer:\n")
            f.write(result + "\n")
            f.write("=" * 60 + "\n")


class GoogleSheetsLogHandler(LogHandler):
    """Handler for logging to Google Sheets."""
    
    def __init__(self):
        try:
            from agentic_student_assistant.core.utils.sheets_logger import log_to_gsheet # pylint: disable=import-outside-toplevel
            self.log_fn = log_to_gsheet
            self.enabled = True
        except ImportError:
            print("⚠️ Google Sheets logging not available")
            self.enabled = False
    
    def log(self, **kwargs):
        """Write log entry to Google Sheets."""
        if not self.enabled:
            return
        
        try:
            self.log_fn(
                timestamp=kwargs.get('timestamp', datetime.datetime.now().isoformat()),
                query=kwargs.get('query', ''),
                agent=kwargs.get('agent', 'unknown'),
                curriculum_mode=kwargs.get('curriculum_mode', 'srh'),
                latency=kwargs.get('latency', 0),
                is_fallback=kwargs.get('is_fallback', False),
                result=kwargs.get('result', '')
            )
        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"⚠️ Google Sheets logging failed: {e}")


class ConsoleLogHandler(LogHandler):
    """Handler for logging to console."""
    
    def log(self, **kwargs):
        """Print log entry to console."""
        query = kwargs.get('query', '')
        agent = kwargs.get('agent', 'unknown')
        confidence = kwargs.get('confidence')
        
        print(f"\n{'='*60}")
        print(f"🔎 Query: {query}")
        print(f"📌 Routed to: {agent}")
        if confidence is not None:
            print(f"🎯 Confidence: {confidence:.2f}")
        print(f"{'='*60}\n")


class LoggingManager:
    """Centralized logging manager supporting multiple handlers."""
    
    def __init__(
        self, 
        log_dir: str = "logs",
        enable_file: bool = True,
        enable_gsheets: bool = False,
        enable_console: bool = False
    ):
        self.handlers: List[LogHandler] = []
        
        if enable_file:
            self.handlers.append(FileLogHandler(log_dir=log_dir))
        
        if enable_gsheets:
            self.handlers.append(GoogleSheetsLogHandler())
        
        if enable_console:
            self.handlers.append(ConsoleLogHandler())
    
    def log_interaction(
        self,
        query: str,
        agent: str,
        result: str,
        latency: Optional[float] = None,
        is_fallback: bool = False,
        curriculum_mode: str = "srh",
        confidence: Optional[float] = None,
        reasoning: str = "",
        **extra_kwargs
    ): # pylint: disable=R0917
        """
        Log an agent interaction.
        
        Args:
            query: User query
            agent: Agent that handled the query
            result: Agent response
            latency: Response time in seconds
            is_fallback: Whether fallback was used
            curriculum_mode: Curriculum mode (srh or uploaded)
            confidence: Router confidence score
            reasoning: Router reasoning
            **extra_kwargs: Additional fields to log
        """
        log_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'query': query,
            'agent': agent,
            'result': result,
            'latency': latency,
            'is_fallback': is_fallback,
            'curriculum_mode': curriculum_mode,
            'confidence': confidence,
            'reasoning': reasoning,
            **extra_kwargs
        }
        
        for handler in self.handlers:
            try:
                handler.log(**log_data)
            except Exception as e: # pylint: disable=broad-exception-caught
                print(f"⚠️ Log handler failed: {e}")
    
    def add_handler(self, handler: LogHandler):
        """Add a custom log handler."""
        self.handlers.append(handler)


if __name__ == "__main__":
    # Test logging
    logger = LoggingManager(enable_file=True, enable_console=True)
    
    logger.log_interaction(
        query="What is machine learning?",
        agent="curriculum",
        result="Machine learning is a subset of AI...",
        latency=1.23,
        confidence=0.95,
        reasoning="Query mentions 'machine learning' which is curriculum content"
    )
    
    print("✅ Test log written to logs/workflow_logs.txt")
