from mlx_lm.models.cache import KVCache, RotatingKVCache
import mlx.nn as nn
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from typing import List, Optional, Any, Dict, Union, Tuple, NamedTuple

class CacheHistory:
    def __init__(self,
            cache: List[Union[KVCache, RotatingKVCache]],
            token_ids: List[List[int]]
        ) -> None:
        self.cache: List[Union[KVCache, RotatingKVCache]] = cache
        self.token_ids: List[List[int]] = token_ids


def make_empty_cache(model: nn.Module, max_kv_size: Optional[int] = None) -> List[Union[KVCache, RotatingKVCache]]:
    """
    Constructs the model's cache for use during generation.

    This function defers the cache construction to the model if it has a
    ``make_cache`` method. If the model does not have this method, a default
    KV cache is created. If a maximum key-value size is specified and the model
    does not have a ``make_cache`` method, a ``RotatingKVCache`` is used with
    the specified maximum size.
    Args:
        model (nn.Module): The language model.
        max_kv_size (Optional[int]): If provided and the model does not have a
            ``make_cache`` method, a ``RotatingKVCache`` is used with a maximum
            size of ``max_kv_size``.

    Returns:
        List[Union[KVCache, RotatingKVCache]]: The constructed cache.
    """
    if hasattr(model, "make_cache"):
        return model.make_cache()

    num_layers = len(model.layers)
    if max_kv_size is not None:
        return [
            RotatingKVCache(max_size=max_kv_size, keep=4) for _ in range(num_layers)
        ]
    else:
        return [KVCache() for _ in range(num_layers)]
    
def create_cache_dict(cache: List[Union[KVCache, RotatingKVCache]]) -> Dict[str, mx.array]:
    """Create a dictionary of prompt cache for saving or further manipuation. 

    Args:
        cache (List[Union[KVCache, RotatingKVCache]]): Prompt cache.

    Returns:
        Dict[str, mx.array]: A dictionary of prompt cache for saving or further manipuation. 
    """
    offset = cache[0].offset
    if offset != 0:
        cache_dict = dict(tree_flatten([c.state for c in cache]))
        return cache_dict
    
def save_cache(cache: List[Union[KVCache, RotatingKVCache]], file: str, metadata: Optional[Dict[str, str]] = None) -> None:
    """Save cache as a safetensors file.

    Args:
        cache (List[Union[KVCache, RotatingKVCache]]): Prompt cache to save.
        file (str): File the cache should be saved.
        metadata (Optional[Dict[str, str]], optional): Metadata for the prompt cache. Defaults to None.
    """
    cache_dict = create_cache_dict(cache=cache)
    if cache_dict:
        mx.save_safetensors(file=file, arrays=cache_dict, metadata=metadata)
    del cache_dict
    mx.metal.clear_cache()

def load_cache(file: str, max_kv_size: Optional[int] = None) -> Tuple[List[Union[KVCache, RotatingKVCache]], Dict[str, str]]:
    """Load a cache file.

    Args:
        file (str): Safetensors file to load.
        max_kv_size (Optional[int], optional): Max number of tokens being kept in the cache. If None given, there is no limitation. Defaults to None.

    Returns:
        Tuple[List[Union[KVCache, RotatingKVCache]], Dict[str, str]]: Cache and it's metadata.
    """
    cache_dict, metadata = mx.load(file, return_metadata=True)
    cache_list = tree_unflatten(list(cache_dict.items()))
    cache = []
    for i, (key, value) in enumerate(cache_list):
        cache.append(RotatingKVCache(max_size=max_kv_size, keep=4) if max_kv_size else KVCache())
        cache[i].update_and_fetch(key, value)
    mx.eval([c.state for c in cache])
    del cache_dict
    mx.metal.clear_cache()
    return cache, metadata

def split_cache(cache: List[Union[KVCache, RotatingKVCache]]) -> List[List[KVCache]]:
    """Split the cache into multiple smaller caches. Each smaller cache will contain a subset of the original cache's data.

    Args:
        cache (List[Union[KVCache, RotatingKVCache]]): The cache to be split.

    Returns:
        List[List[KVCache]]: A list of smaller caches.
    """
    num_split = cache[0].state[0].shape[0]
    caches = []
    def make_cache(n):
        ec = []
        for j, c in enumerate(cache):
            ec.append(KVCache())
            ec[j].update_and_fetch(c.state[0][n:n + 1], c.state[1][n:n + 1])
            mx.eval(ec[j].state)
        return ec
    for i in range(num_split):
        caches.append(make_cache(i))
    del cache
    mx.metal.clear_cache()
    return caches

def trim_cache(cache: List[KVCache], offset: int, trim_suffix: int = 0) -> List[KVCache]:
    """
    Trim the cache by removing entries before the specified offset and optionally
    removing entries after the specified trim_suffix.

    Args:
        cache (List[KVCache]): The cache to be trimmed.
        offset (int): The offset from which to start trimming.
        trim_suffix (int, optional): The number of entries after the offset to keep. Defaults to 0.

    Returns:
        List[KVCache]: The trimmed cache.
    """
    new_cache = []
    for j, c in enumerate(cache):
        new_cache.append(KVCache())
        if trim_suffix == 0:
            new_cache[j].update_and_fetch(c.state[0][:, :, offset:, :], c.state[1][:, :, offset:, :])
        else:
            new_cache[j].update_and_fetch(c.state[0][:, :, offset:-trim_suffix, :], c.state[1][:, :, offset:-trim_suffix, :])
        mx.eval(new_cache[j].state)
    del cache
    mx.metal.clear_cache()
    return new_cache


def select_from_cache(
        cache: List[Union[KVCache, RotatingKVCache]],
        prompt_index: int = 0,
        select_len: Optional[int] = None,
        start_from: int = 0
    ) -> List[Tuple[mx.array, mx.array]]:
    """Get the kv cache tensors of a specific sequence in a cache with multiple prompt cache sequences.

    Args:
        cache (List[Union[KVCache, RotatingKVCache]]): Cache object.
        prompt_index (int, optional): The index of the sequence to get. Defaults to 0.
        select_len (Optional[int], optional): Length of the seqence to get. Defaults to None.
        start_from (int, optional): Starting index of the sequence. Defaults to 0.

    Returns:
        List[Tuple[mx.array, mx.array]]: List of keys and values tuples for updating the new cache.
    """
    num_prompts, num_kv_heads, prompt_len, head_dim = cache[0].state[0].shape
    max_index = select_len + start_from if select_len else prompt_len
    if prompt_index >= num_prompts:
        raise ValueError('Prompt index out of index.')
    if max_index > prompt_len:
        raise ValueError(f'Select length out of index.')
    cache_list = []
    for c in cache:
        k, v  = c.state
        cache_list.append((k[prompt_index:prompt_index + 1, :, start_from:max_index, :], v[prompt_index:prompt_index + 1, :, start_from:max_index, :]))
    return cache_list

def find_max_prefix_num(new: List[int], baseline: List[int]) -> int:
    """Helper function to find the maximum number of tokens shared in the prefix of two prompts.

    Args:
        new (List[int]): First prompt token ids.
        baseline (List[int]): Second prompt token ids.

    Returns:
        int: The maximum number of tokens shared in the prefix of two prompts.
    """
    from itertools import takewhile
    return len(list(takewhile(lambda x: x[0] == x[1], zip(new, baseline))))
