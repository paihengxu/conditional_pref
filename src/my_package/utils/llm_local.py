import os
import logging
import time
from typing import List, Optional

logger = logging.getLogger(__name__)

# vLLM lazy loading engines cache
_LOCAL_ENGINES = {}


def _sleep_all_except(active_model: Optional[str] = None) -> None:
    """Put every cached vLLM engine *except* `active` to sleep to free up GPU memory."""
    for name, engine in _LOCAL_ENGINES.items():
        if name == active_model:
            continue
        if engine.llm_engine.is_sleeping():
            continue
        logger.info(f"Sleeping local engine '{name}' to free GPU memory...")
        engine.llm_engine.reset_prefix_cache()
        # Level 1 clears KV cache and moves weights to CPU; Level 2 completely clears weights
        engine.sleep(level=2)


def get_vllm_engine(model: str, tensor_parallel_size: int = 1, **kwargs):
    """
    Return or initialize a vLLM engine for the specified model.

    * If the engine is already cached, sleeps other models and wakes it up.
    * If not cached, sleeps other engines to clear GPU memory, then loads it.

    Args:
        model: HuggingFace model hub path or local path
        tensor_parallel_size: Number of GPUs to use (tensor parallelism)
        **kwargs: Additional engine arguments passed to vLLM's LLM constructor
    """
    from vllm import LLM
    import torch

    # Ensure optimal TensorFloat32 settings
    torch.set_float32_matmul_precision("high")

    engine = _LOCAL_ENGINES.get(model)

    if engine is None:
        _sleep_all_except(active_model=None)  # Free GPU before allocating

        logger.info(f"Loading local model '{model}' via vLLM across {tensor_parallel_size} GPU(s)...")
        t0 = time.time()
        
        # Pop standard config parameters with defaults
        gpu_memory_utilization = kwargs.pop("gpu_memory_utilization", 0.9)
        task = kwargs.pop("task", "generate")
        enable_sleep_mode = kwargs.pop("enable_sleep_mode", True)

        engine = LLM(
            model=model,
            task=task,
            enable_sleep_mode=enable_sleep_mode,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            **kwargs
        )
        _LOCAL_ENGINES[model] = engine
        
        dtype = getattr(engine.llm_engine.get_model_config(), "dtype", "unknown")
        logger.info(f"Successfully loaded '{model}' with dtype: {dtype} (took {time.time()-t0:.1f}s)")
    else:
        _sleep_all_except(active_model=model)
        if engine.llm_engine.is_sleeping():
            logger.info(f"Engine found for '{model}' but is currently sleeping. Waking up...")
            logger.warning("Note: waking up models in some vLLM versions may cause output anomalies.")
            engine.wake_up()
            engine.llm_engine.reset_prefix_cache()

    return engine


def get_local_completions(
    prompts: List[str],
    model: str = "Qwen/Qwen3-0.6B",
    max_tokens: int = 128,
    show_progress: bool = True,
    tokenizer_kwargs: Optional[dict] = None,
    llm_sampling_kwargs: Optional[dict] = None,
    vllm_engine_kwargs: Optional[dict] = None,
) -> List[str]:
    """
    Generate completions using a local vLLM model.

    Args:
        prompts: List of raw prompts to feed into the model
        model: HuggingFace model hub path or local path
        max_tokens: Maximum tokens to generate per completion
        show_progress: If True, display a progress bar
        tokenizer_kwargs: Dictionary of additional tokenizer options
        llm_sampling_kwargs: Dictionary of sampling parameters (e.g. temperature)
        vllm_engine_kwargs: Dictionary of configurations passed directly to LLM() constructor
                            (e.g., {"tensor_parallel_size": 2} to run on 2 GPUs)

    Returns:
        List of generated completion strings in the same order as prompts
    """
    from vllm import SamplingParams

    tokenizer_kwargs = tokenizer_kwargs or {}
    llm_sampling_kwargs = llm_sampling_kwargs or {}
    vllm_engine_kwargs = vllm_engine_kwargs or {}

    # Extract or default tensor_parallel_size (number of GPUs)
    tensor_parallel_size = vllm_engine_kwargs.pop("tensor_parallel_size", 1)

    # Load or fetch the engine
    engine = get_vllm_engine(model, tensor_parallel_size=tensor_parallel_size, **vllm_engine_kwargs)
    tokenizer = engine.get_tokenizer()

    # Automatically format prompts into chat templates if supported
    if getattr(tokenizer, "chat_template", None) is not None:
        messages_lists = [[{"role": "user", "content": p}] for p in prompts]
        enable_thinking = tokenizer_kwargs.pop("enable_thinking", False)
        
        prompts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                **tokenizer_kwargs
            )
            for messages in messages_lists
        ]

    sampling_params = SamplingParams(max_tokens=max_tokens, **llm_sampling_kwargs)
    outputs = engine.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=show_progress,
    )

    completions = [str(out.outputs[0].text) for out in outputs]
    return completions
