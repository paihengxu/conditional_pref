import os
import json
import hashlib
import logging
import threading
from pathlib import Path
from typing import Optional, Callable, Any, Dict

logger = logging.getLogger(__name__)

# Default completion cache directory (4 levels up from my_package/utils/cache.py -> project root)
DEFAULT_CACHE_DIR = os.getenv('COMPLETION_CACHE_DIR') or os.path.join(
    Path(__file__).parent.parent.parent.parent, 'completion_cache'
)


class CompletionCache:
    """
    A simple, thread-safe, file-based key-value cache for LLM completions.
    Speeds up development, facilitates regression testing, and saves API costs.
    """
    def __init__(self, cache_name: str = "completions_cache", cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, f"{cache_name}.jsonl")
        
        self.lock = threading.Lock()
        self.cache: Dict[str, str] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache entries from the JSONL file."""
        with self.lock:
            if not os.path.exists(self.cache_path):
                return
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entry = json.loads(line)
                            self.cache[entry['key']] = entry['value']
                logger.info(f"Loaded {len(self.cache)} cached completions from {self.cache_path}")
            except Exception as e:
                logger.warning(f"Failed to read completion cache file {self.cache_path}: {e}. Starting fresh.")

    def _generate_key(self, prompt: str, model: str, **kwargs) -> str:
        """Generate a deterministic, secure SHA256 key from request inputs."""
        # Normalize keyword arguments
        stable_kwargs = json.dumps(kwargs, sort_keys=True)
        payload = f"model:{model}|||prompt:{prompt}|||kwargs:{stable_kwargs}"
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def get(self, prompt: str, model: str, **kwargs) -> Optional[str]:
        """Retrieve a cached completion if it exists."""
        key = self._generate_key(prompt, model, **kwargs)
        with self.lock:
            return self.cache.get(key)

    def set(self, prompt: str, model: str, completion: str, **kwargs) -> None:
        """Cache a completion and write it atomically to the JSONL file."""
        key = self._generate_key(prompt, model, **kwargs)
        
        with self.lock:
            if key in self.cache:
                return  # Already cached
            
            self.cache[key] = completion
            
            # Append entry to JSONL file
            try:
                with open(self.cache_path, 'a', encoding='utf-8') as f:
                    json.dump({'key': key, 'value': completion, 'prompt_preview': prompt[:100]}, f, ensure_ascii=False)
                    f.write('\n')
            except Exception as e:
                logger.error(f"Failed to write to cache file {self.cache_path}: {e}")


def cached_completion(cache_name: str = "completions_cache", cache_dir: Optional[str] = None):
    """
    Decorator for caching completions.
    
    The decorated function MUST take 'prompt' and 'model' as its first arguments or keyword arguments.
    """
    cache = CompletionCache(cache_name=cache_name, cache_dir=cache_dir)

    def decorator(func: Callable[..., str]):
        def wrapper(prompt: str, model: str = "default", *args, **kwargs) -> str:
            # Check cache
            cached_val = cache.get(prompt, model, **kwargs)
            if cached_val is not None:
                return cached_val
            
            # Execute completion
            result = func(prompt, model, *args, **kwargs)
            
            # Cache the new result
            if result:
                cache.set(prompt, model, result, **kwargs)
            
            return result
        return wrapper
    return decorator
