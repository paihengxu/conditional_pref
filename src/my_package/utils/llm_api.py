import os
import time
import logging
import glob
import concurrent.futures
from typing import List, Optional, Dict
from pathlib import Path
import numpy as np
import openai
import dotenv
from tqdm.auto import tqdm

from my_package.utils.cache import CompletionCache

logger = logging.getLogger(__name__)

# Model Mapping: Abbrev -> Standard API Model ID
model_abbrev_to_id = {
    'gpt4o': 'gpt-4o-2024-11-20',
    'gpt-4o': 'gpt-4o-2024-11-20',
    'gpt4o-mini': 'gpt-4o-mini-2024-07-18',
    'gpt-4o-mini': 'gpt-4o-mini-2024-07-18',
    "gpt4.1": "gpt-4.1-2025-04-14",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt4.1-nano": "gpt-4.1-nano-2025-04-14",
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
    "gpt5": "gpt-5-2025-08-07",
    "gpt-5": "gpt-5-2025-08-07",
    "gpt-5.4": "gpt-5.4-2026-03-05",
    "gpt5.4": "gpt-5.4-2026-03-05",
}

DEFAULT_MODEL = "gpt-4o-mini"

# Global lazy-loaded OpenAI client
_CLIENT_OPENAI: Optional[openai.OpenAI] = None

# Global lazy-loaded Completion cache
_COMPLETION_CACHE: Optional[CompletionCache] = None

# Embedding Cache directory (4 levels up from my_package/utils/llm_api.py -> project root)
EMB_CACHE_DIR = os.getenv('EMB_CACHE_DIR') or os.path.join(
    Path(__file__).parent.parent.parent.parent, 'emb_cache'
)


def get_client() -> openai.OpenAI:
    """Get the OpenAI client, initializing it if necessary and caching it."""
    global _CLIENT_OPENAI
    if _CLIENT_OPENAI is not None:
        return _CLIENT_OPENAI

    # Find and load the nearest .env file
    dotenv.load_dotenv(dotenv.find_dotenv())
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("openai_api_key")
    if api_key is None or '...' in api_key:
        raise ValueError(
            "Please set the OPENAI_API_KEY environment variable before using "
            "functions which require the OpenAI API."
        )

    # Use default/custom base URL if provided in environment
    base_url = os.getenv("OPENAI_BASE_URL") or "https://us.api.openai.com/v1"
    _CLIENT_OPENAI = openai.OpenAI(api_key=api_key, base_url=base_url)
    return _CLIENT_OPENAI


def get_completion_cache() -> CompletionCache:
    """Get or initialize the global completion cache."""
    global _COMPLETION_CACHE
    if _COMPLETION_CACHE is None:
        _COMPLETION_CACHE = CompletionCache()
    return _COMPLETION_CACHE


def get_completion(
    prompt: str,
    model: str = DEFAULT_MODEL,
    use_cache: bool = True,
    timeout: float = 15.0,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    **kwargs
) -> str:
    """
    Get completion from OpenAI API with retry logic, timeout, and optional caching.

    Args:
        prompt: The prompt to send
        model: Model alias or full model ID to use
        use_cache: If True, check the local completion cache before querying the API
        timeout: Timeout for the request in seconds
        max_retries: Maximum number of retries on API timeout/rate-limits
        backoff_factor: Factor to multiply wait time by after each retry
        **kwargs: Additional arguments to pass to OpenAI (e.g. temperature, max_tokens)

    Returns:
        Generated completion text string
    """
    model_id = model_abbrev_to_id.get(model, model)

    # Check cache if requested
    if use_cache:
        cache = get_completion_cache()
        cached_result = cache.get(prompt, model_id, **kwargs)
        if cached_result is not None:
            return cached_result

    client = get_client()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
                **kwargs
            )
            result = response.choices[0].message.content
            
            # Save to cache if successful
            if use_cache and result:
                cache.set(prompt, model_id, result, **kwargs)
            
            return result

        except (openai.RateLimitError, openai.APITimeoutError) as e:
            if attempt == max_retries - 1:  # Last attempt
                raise e

            wait_time = timeout * (backoff_factor ** attempt)
            logger.warning(
                f"OpenAI API error: {e}; retrying in {wait_time:.1f}s... "
                f"({attempt + 1}/{max_retries})"
            )
            time.sleep(wait_time)


# --- Embedding Helpers with Chunked Disk Caching ---

def _embed_batch_openai(
    batch: List[str],
    model: str,
    client: openai.OpenAI,
    max_tokens: int = 8192,
    max_retries: int = 3,
    backoff_factor: float = 3.0,
    timeout: float = 10.0
) -> List[List[float]]:
    """Helper function for batch embedding using OpenAI API with local token truncation."""
    try:
        import tiktoken
    except ImportError as e:
        raise ImportError(
            "Batch embedding truncation requires 'tiktoken' package.\n"
            "Please install it using:\n"
            "  pip install tiktoken"
        ) from e
    enc = tiktoken.get_encoding("cl100k_base")
    truncated_batch = []
    for text in batch:
        tokens = enc.encode(text.strip())
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            text = enc.decode(tokens)
        truncated_batch.append(text)

    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                input=truncated_batch,
                model=model,
                timeout=timeout
            )
            return [data.embedding for data in response.data]

        except (openai.RateLimitError, openai.APITimeoutError) as e:
            if attempt == max_retries - 1:
                raise e

            wait_time = timeout * (backoff_factor ** attempt)
            logger.warning(f"API error: {e}; retrying in {wait_time:.1f}s... ({attempt + 1}/{max_retries})")
            time.sleep(wait_time)


def load_embedding_cache(cache_name: str) -> dict:
    """Load cached embeddings from chunked files on disk."""
    if not cache_name:
        return {}

    cache_dir = os.path.join(EMB_CACHE_DIR, cache_name)
    if not os.path.exists(cache_dir):
        return {}

    text2embedding = {}
    chunk_files = sorted(glob.glob(os.path.join(cache_dir, "chunk_*.npy")))

    for chunk_file in tqdm(chunk_files, desc="Loading embedding chunks", leave=False):
        chunk_data = np.load(chunk_file, allow_pickle=True)
        for text, emb in chunk_data:
            text2embedding[text] = emb

    logger.info(f"Loaded {len(text2embedding)} cached embeddings from '{cache_name}'")
    return text2embedding


def _save_embedding_chunk(cache_name: str, chunk_embeddings: dict, chunk_idx: int) -> int:
    """Save a chunk of embeddings to disk."""
    if not cache_name or not chunk_embeddings:
        return chunk_idx

    cache_dir = os.path.join(EMB_CACHE_DIR, cache_name)
    os.makedirs(cache_dir, exist_ok=True)

    chunk_path = os.path.join(cache_dir, f"chunk_{chunk_idx:03d}.npy")
    chunk_items = list(chunk_embeddings.items())
    np.save(chunk_path, np.array(chunk_items, dtype=object))
    logger.info(f"Saved {len(chunk_items)} embeddings to {chunk_path}")

    return chunk_idx + 1


def _get_next_chunk_index(cache_name: str) -> int:
    """Determine the next available chunk index for a cache folder."""
    if not cache_name:
        return 0

    cache_dir = os.path.join(EMB_CACHE_DIR, cache_name)
    if not os.path.exists(cache_dir):
        return 0

    chunk_files = glob.glob(os.path.join(cache_dir, "chunk_*.npy"))
    if not chunk_files:
        return 0

    indices = [int(os.path.basename(f).split("_")[1].split(".")[0]) for f in chunk_files]
    return max(indices) + 1


def get_openai_embeddings(
    texts: List[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 256,
    n_workers: int = 5,
    cache_name: Optional[str] = None,
    show_progress: bool = True,
    chunk_size: int = 50000,
    timeout: float = 10.0,
) -> Dict[str, np.ndarray]:
    """Get embeddings using OpenAI API with parallel processing and chunked caching."""
    # Setup cache
    text2embedding = load_embedding_cache(cache_name) if cache_name else {}
    texts_to_embed = [text for text in texts if text not in text2embedding]

    if not texts_to_embed:
        return text2embedding

    client = get_client()
    next_chunk_idx = _get_next_chunk_index(cache_name) if cache_name else 0

    # Create chunk ranges
    chunk_ranges = [(i, min(i + chunk_size, len(texts_to_embed)))
                    for i in range(0, len(texts_to_embed), chunk_size)]

    chunk_iterator = chunk_ranges
    if show_progress:
        chunk_iterator = tqdm(chunk_iterator, desc="Processing chunks", total=len(chunk_ranges))

    for chunk_start, chunk_end in chunk_iterator:
        chunk_texts = texts_to_embed[chunk_start:chunk_end]
        chunk_embeddings = {}

        # Process chunk in batches with parallel workers
        batches = [chunk_texts[i:i + batch_size] for i in range(0, len(chunk_texts), batch_size)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for batch in batches:
                futures.append(executor.submit(_embed_batch_openai, batch, model, client, timeout=timeout))

            iterator = concurrent.futures.as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(batches), desc=f"Embedding Chunk {next_chunk_idx}", leave=False)

            for future in iterator:
                batch_result = future.result()
                batch_idx = futures.index(future)
                batch = batches[batch_idx]

                for text, embedding in zip(batch, batch_result):
                    chunk_embeddings[text] = embedding
                    text2embedding[text] = embedding

        # Save completed chunk
        if cache_name:
            next_chunk_idx = _save_embedding_chunk(cache_name, chunk_embeddings, next_chunk_idx)

    return text2embedding
