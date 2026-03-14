"""
Feedback storage and analytics using Redis.
Stores user feedback (thumbs up/down) for agent responses.
"""
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import redis
from dotenv import load_dotenv

load_dotenv()


class FeedbackStore:
    """Manages user feedback storage in Redis."""
    
    def __init__(self):
        """Initialize Redis connection for feedback storage."""
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD"),
            decode_responses=True
        )
        
    def add_feedback(
        self,
        query: str,
        response: str,
        agent: str,
        rating: int,
        session_id: str,
        latency: Optional[float] = None,
        confidence: Optional[float] = None,
        comment: Optional[str] = None
    ) -> str:
        """
        Store user feedback in Redis.
        
        Args:
            query: User's query
            response: Agent's response
            agent: Agent that handled the query
            rating: 1 for positive (👍), -1 for negative (👎)
            session_id: User session identifier
            latency: Response time in seconds
            confidence: Router confidence score
            comment: Optional user comment
            
        Returns:
            Feedback ID (timestamp-based)
        """
        timestamp = datetime.utcnow().isoformat()
        feedback_id = f"{timestamp}:{session_id[:8]}"
        
        feedback_data = {
            "id": feedback_id,
            "timestamp": timestamp,
            "query": query,
            "response": response[:500],  # Truncate long responses
            "agent": agent,
            "rating": rating,
            "session_id": session_id,
            "latency": latency,
            "confidence": confidence,
            "comment": comment
        }
        
        # Store in agent-specific list
        self.redis_client.lpush(
            f"feedback:{agent}",
            json.dumps(feedback_data)
        )
        
        # Also store in global list for overall stats
        self.redis_client.lpush(
            "feedback:all",
            json.dumps(feedback_data)
        )
        
        # Set expiration (keep for 90 days)
        self.redis_client.expire(f"feedback:{agent}", 90 * 24 * 60 * 60)
        self.redis_client.expire("feedback:all", 90 * 24 * 60 * 60)
        
        return feedback_id
    
    def get_agent_feedback(
        self,
        agent: str,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get feedback for a specific agent.
        
        Args:
            agent: Agent name
            limit: Maximum number of feedback entries to return
            
        Returns:
            List of feedback dictionaries
        """
        feedback_json = self.redis_client.lrange(f"feedback:{agent}", 0, limit - 1)
        return [json.loads(f) for f in feedback_json]
    
    def get_all_feedback(self, limit: int = 100) -> List[Dict]:
        """
        Get all feedback across all agents.
        
        Args:
            limit: Maximum number of feedback entries to return
            
        Returns:
            List of feedback dictionaries
        """
        feedback_json = self.redis_client.lrange("feedback:all", 0, limit - 1)
        return [json.loads(f) for f in feedback_json]
    
    def get_satisfaction_rate(self, agent: Optional[str] = None) -> float:
        """
        Calculate satisfaction rate (percentage of positive feedback).
        
        Args:
            agent: Specific agent name, or None for overall rate
            
        Returns:
            Satisfaction rate as percentage (0-100)
        """
        if agent:
            feedback = self.get_agent_feedback(agent, limit=-1)
        else:
            feedback = self.get_all_feedback(limit=-1)
        
        if not feedback:
            return 0.0
        
        positive = sum(1 for f in feedback if f["rating"] == 1)
        total = len(feedback)
        
        return (positive / total) * 100 if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Dict]:
        """
        Get comprehensive statistics for all agents.
        
        Returns:
            Dictionary with stats per agent and overall
        """
        # Get all unique agents
        all_feedback = self.get_all_feedback(limit=-1)
        agents = set(f["agent"] for f in all_feedback)
        
        stats = {}
        
        # Per-agent stats
        for agent in agents:
            feedback = self.get_agent_feedback(agent, limit=-1)
            positive = sum(1 for f in feedback if f["rating"] == 1)
            negative = sum(1 for f in feedback if f["rating"] == -1)
            total = len(feedback)
            
            stats[agent] = {
                "total": total,
                "positive": positive,
                "negative": negative,
                "satisfaction_rate": (positive / total * 100) if total > 0 else 0.0,
                "avg_latency": sum(f.get("latency", 0) or 0 for f in feedback) / total if total > 0 else 0.0,
                "avg_confidence": sum(f.get("confidence", 0) or 0 for f in feedback) / total if total > 0 else 0.0
            }
        
        # Overall stats
        positive_all = sum(1 for f in all_feedback if f["rating"] == 1)
        negative_all = sum(1 for f in all_feedback if f["rating"] == -1)
        total_all = len(all_feedback)
        
        stats["overall"] = {
            "total": total_all,
            "positive": positive_all,
            "negative": negative_all,
            "satisfaction_rate": (positive_all / total_all * 100) if total_all > 0 else 0.0
        }
        
        return stats
    
    def get_low_rated_queries(
        self,
        agent: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get queries with negative feedback for improvement analysis.
        
        Args:
            agent: Specific agent name, or None for all agents
            limit: Maximum number of queries to return
            
        Returns:
            List of feedback entries with negative ratings
        """
        if agent:
            feedback = self.get_agent_feedback(agent, limit=-1)
        else:
            feedback = self.get_all_feedback(limit=-1)
        
        # Filter negative feedback
        negative_feedback = [f for f in feedback if f["rating"] == -1]
        
        # Sort by timestamp (most recent first)
        negative_feedback.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return negative_feedback[:limit]
    
    def clear_agent_feedback(self, agent: str) -> bool:
        """
        Clear all feedback for a specific agent.
        
        Args:
            agent: Agent name
            
        Returns:
            True if successful
        """
        return self.redis_client.delete(f"feedback:{agent}") > 0
    
    def clear_all_feedback(self) -> bool:
        """
        Clear all feedback data.
        
        Returns:
            True if successful
        """
        # Get all feedback keys
        keys = self.redis_client.keys("feedback:*")
        if keys:
            return self.redis_client.delete(*keys) > 0
        return True


# Singleton instance
_feedback_store = None

def get_feedback_store() -> FeedbackStore:
    """Get or create FeedbackStore singleton instance."""
    global _feedback_store
    if _feedback_store is None:
        _feedback_store = FeedbackStore()
    return _feedback_store
