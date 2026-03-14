import time
import hashlib
import json
import os
import numpy as np
from typing import Optional, Any, Dict
from collections import OrderedDict
from huggingface_hub import snapshot_download
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"

# Optional redis and openai imports
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class ResponseCache:
    """
    Standard exact-match LRU cache (In-memory).
    """
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, query: str) -> str:
        normalized = f"query_cache:{query.lower().strip()}"
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get(self, query: str, agent: str = "") -> Optional[str]:
        key = self._generate_key(query)
        if key not in self.cache:
            self.misses += 1
            return None
        
        entry = self.cache[key]
        if time.time() - entry['timestamp'] > self.ttl_seconds:
            del self.cache[key]
            self.misses += 1
            return None
        
        self.cache.move_to_end(key)
        self.hits += 1
        return entry['response']
    
    def set(self, query: str, response: str, agent: str = ""):
        key = self._generate_key(query)
        if len(self.cache) >= self.max_size and key not in self.cache:
            self.cache.popitem(last=False)
        self.cache[key] = {
            'response': response,
            'timestamp': time.time()
        }
    
    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            'hits': self.hits,
            'misses': self.misses,
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': self.hits / total if total > 0 else 0,
            'type': 'in-memory'
        }


class SemanticRedisCache:
    """
    Persistent cache using Redis with Local Semantic Similarity support.
    Uses SentenceTransformers (locally) to find 'similar' questions.
    """
    
    # Singleton for model loading to avoid re-loading on every call
    _model = None
    _model_ready = False

    def __init__(
        self,
        host='localhost',
        port=6379,
        db=0,
        password=None,
        ttl_seconds=3600,
        similarity_threshold=0.88,
        max_size: int = 1000,
        **redis_kwargs
    ): # pylint: disable=R0917
        self.ttl_seconds = ttl_seconds
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.client = redis.Redis(
            host=host, port=port, db=db, password=password, 
            decode_responses=True, socket_timeout=5.0,
            **redis_kwargs
        )
        self.hits = 0
        self.misses = 0
        
        # Test connection
        self.client.ping()
        
        # Load local model lazily
        self._load_model()

    def _load_model(self):
        if SemanticRedisCache._model_ready:
            return

        model_path = Path.home() / ".cache" / "huggingface" / "hub"

        print(f"[INFO] Checking embedding model: {MODEL_NAME}")

        try:
            # Explicit check: is it already cached?
            local_models = list(model_path.glob(f"**/*{MODEL_NAME.replace('/', '--')}*"))
            if not local_models:
                print("[INFO] Model not found locally. Downloading…")
                snapshot_download(repo_id=MODEL_NAME)
                print("[SUCCESS] Model downloaded.")

            # Load after confirmation
            SemanticRedisCache._model = SentenceTransformer(MODEL_NAME)
            SemanticRedisCache._model_ready = True
            print("[INFO] Embedding model ready.")

        except Exception as e:
            SemanticRedisCache._model = None
            SemanticRedisCache._model_ready = False
            raise RuntimeError(
                f"Failed to initialize embedding model '{MODEL_NAME}'. "
                f"Semantic cache disabled."
            ) from e


    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        if not SemanticRedisCache._model:
            return None
        try:
            return SemanticRedisCache._model.encode(text.lower().strip())
        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"[ERROR] Local embedding error: {e}")
            return None

    def _cosine_similarity(self, v1, v2):
        v1, v2 = np.array(v1), np.array(v2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def get(self, query: str, agent: str = "") -> Optional[str]:
        # 1. Try EXACT match first (Fastest)
        q_clean = query.lower().strip()
        exact_key = f"exact_cache:{hashlib.md5(q_clean.encode()).hexdigest()}"
        try:
            exact_data = self.client.get(exact_key)
            if exact_data:
                self.hits += 1
                return exact_data
        except Exception: # pylint: disable=broad-exception-caught
            pass

        # 2. Try LOCAL SEMANTIC match
        if not SENTENCE_TRANSFORMERS_AVAILABLE or not SemanticRedisCache._model:
            self.misses += 1
            return None

        query_emb = self._get_embedding(q_clean)
        if query_emb is None:
            self.misses += 1
            return None

        try:
            # Note: For production use with millions of items, 
            # you'd use Redis VSS indexes. For a few thousand, this scan is fast.
            recent_keys = self.client.keys("semantic_meta:*")
            for meta_key in recent_keys:
                meta_json = self.client.get(meta_key)
                if not meta_json:
                    continue
                
                meta = json.loads(meta_json)
                similarity = self._cosine_similarity(query_emb, meta['embedding'])
                print(f"[DEBUG] Similarity with '{meta.get('query')}': {similarity:.4f}")
                
                if similarity >= self.threshold:
                    print(f"[SUCCESS] Semantic Hit! Similarity: {similarity:.2f} (Match: '{meta['query']}')")
                    self.hits += 1
                    return self.client.get(meta['payload_key'])
            
            self.misses += 1
            return None
        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"[ERROR] Redis Semantic Search Error: {e}")
            self.misses += 1
            return None

    def set(self, query: str, response: str, agent: str = ""):
        query_clean = query.lower().strip()
        q_hash = hashlib.md5(query_clean.encode()).hexdigest()
        
        exact_key = f"exact_cache:{q_hash}"
        payload_key = f"payload_cache:{q_hash}"
        meta_key = f"semantic_meta:{q_hash}"

        try:
            # Store payload
            self.client.setex(payload_key, self.ttl_seconds, response)
            # Store exact match pointer
            self.client.setex(exact_key, self.ttl_seconds, response)
            
            # Store semantic metadata locally
            if SENTENCE_TRANSFORMERS_AVAILABLE and SemanticRedisCache._model:
                emb = self._get_embedding(query_clean)
                if emb is not None:
                    meta = {
                        "query": query_clean,
                        "embedding": emb.tolist(),
                        "payload_key": payload_key
                    }
                    self.client.setex(meta_key, self.ttl_seconds, json.dumps(meta))
        except Exception as e: # pylint: disable=broad-exception-caught
            print(f"[ERROR] Redis SET Error: {e}")

    def clear(self):
        try:
            keys = self.client.keys("*_cache:*") + self.client.keys("semantic_meta:*")
            if keys:
                self.client.delete(*keys)
            self.hits = 0
            self.misses = 0
        except Exception: # pylint: disable=broad-exception-caught
            pass

    def get_stats(self) -> Dict[str, Any]:
        try:
            total = self.hits + self.misses
            # In Redis, we count payload keys. This might be more than max_size if TTL 
            # is long, but for stats we show current count.
            size = len(self.client.keys("payload_cache:*"))
            return {
                'hits': self.hits,
                'misses': self.misses,
                'size': size,
                'max_size': self.max_size,
                'hit_rate': self.hits / total if total > 0 else 0,
                'type': 'redis-semantic'
            }
        except Exception: # pylint: disable=broad-exception-caught
            return {
                'hits': self.hits,
                'misses': self.misses,
                'size': 0,
                'max_size': self.max_size,
                'hit_rate': 0,
                'type': 'redis-fail'
            }


# Global cache instance
_global_cache: Any = None

def get_cache(ttl_seconds: int = 3600, max_size: int = 1000) -> Any:
    global _global_cache  # pylint: disable=global-statement
    if _global_cache is not None:
        return _global_cache

    if REDIS_AVAILABLE:
        try:
            host = os.getenv("REDIS_HOST", "localhost")
            port = int(os.getenv("REDIS_PORT", "6379"))
            db = int(os.getenv("REDIS_DB", "0"))
            password = os.getenv("REDIS_PASSWORD")
            
            print(f"[INFO] Connecting to Semantic Redis at {host}...")
            _global_cache = SemanticRedisCache(
                host=host, port=port, db=db, password=password, 
                ttl_seconds=ttl_seconds, max_size=max_size
            )
            return _global_cache
        except Exception as redis_err: # pylint: disable=broad-exception-caught
            print(f"⚠️ Redis failed: {redis_err}. Falling back to In-memory.")
    
    _global_cache = ResponseCache(ttl_seconds=ttl_seconds, max_size=max_size)
    return _global_cache


if __name__ == "__main__":
    # Test
    c = get_cache()
    print(f"Running {c.get_stats()['type']}")
    c.set("What are the best AI jobs?", "The best AI jobs are...")
    print(f"Similar match: {c.get('What are best jobs in AI?')}")
