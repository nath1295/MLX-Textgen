# Adapted from mlx-lm

import mlx.nn as nn
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten, tree_map
from typing import List, Optional, Any, Dict, Union, Tuple, NamedTuple

class _BaseCache:
    @property
    def state(self):
        return []

    @state.setter
    def state(self, v):
        if v is not None and v:
            raise ValueError("This cache has no state but a state was set.")

    @property
    def meta_state(self):
        return ""

    @meta_state.setter
    def meta_state(self, v):
        if v is not None and v:
            raise ValueError("This cache has no meta_state but a meta_state was set.")

    def is_trimmable(self):
        return False


class QuantizedKVCache(_BaseCache):
    def __init__(self, group_size: int = 64, bits: int = 8):
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = 256
        self.group_size = group_size
        self.bits = bits

    def update_and_fetch(self, keys, values):
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self.offset

        if self.keys is None or (prev + num_steps) > self.keys[0].shape[-2]:
            el_per_int = 8 * mx.uint32.size // self.bits
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            shape = (B, n_kv_heads, new_steps)

            def init_quant(dim):
                return (
                    mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                )

            def expand_quant(x):
                new_x = mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)
                return mx.concatenate([x, new_x], axis=-2)

            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys, self.values = tree_map(
                        lambda x: x[..., :prev, :], (self.keys, self.values)
                    )

                self.keys, self.values = tree_map(
                    expand_quant, (self.keys, self.values)
                )
            else:
                self.keys, self.values = init_quant(k_head_dim), init_quant(v_head_dim)

        self.offset += num_steps

        keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        values = mx.quantize(values, group_size=self.group_size, bits=self.bits)
        for i in range(len(self.keys)):
            self.keys[i][..., prev : self.offset, :] = keys[i]
            self.values[i][..., prev : self.offset, :] = values[i]

        return tree_map(lambda x: x[..., : self.offset, :], (self.keys, self.values))

    @property
    def state(self):
        if self.offset == self.keys[0].shape[2]:
            return self.keys, self.values
        else:
            return tree_map(
                lambda x: x[..., : self.offset, :], (self.keys, self.values)
            )

    @state.setter
    def state(self, v):
        self.keys, self.values = v

    @property
    def meta_state(self):
        return tuple(map(str, (self.step, self.offset, self.group_size, self.bits)))

    @meta_state.setter
    def meta_state(self, v):
        self.step, self.offset, self.group_size, self.bits = map(int, v)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n


class KVCache(_BaseCache):
    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = 256

    def update_and_fetch(self, keys, values):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    @property
    def state(self):
        if self.offset == self.keys.shape[2]:
            return self.keys, self.values
        else:
            return (
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            )

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = self.keys.shape[2]

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def to_quantized(self, group_size: int = 64, bits: int = 4) -> QuantizedKVCache:
        quant_cache = QuantizedKVCache(group_size=group_size, bits=bits)
        quant_cache.offset = self.offset
        if self.keys is not None:
            quant_cache.keys = mx.quantize(self.keys, group_size=group_size, bits=bits)
            quant_cache.values = mx.quantize(
                self.values, group_size=group_size, bits=bits
            )
        return quant_cache


class CacheHistory:
    def __init__(self,
            cache: List[Union[KVCache, QuantizedKVCache]],
            token_ids: List[List[int]]
        ) -> None:
        self.cache: List[Union[KVCache, QuantizedKVCache]] = cache
        self.token_ids: List[List[int]] = token_ids


def make_empty_cache(model: nn.Module) -> List[Union[KVCache, QuantizedKVCache]]:
    """
    Constructs an empty cache for use during generation.

    This function creates an empty cache for a given language model. If the model
    has a ``make_cache`` method, it defers the cache construction to the model.
    Otherwise, it creates a default KV cache.
    Args:
        model (nn.Module): The language model.
    Returns:
        List[Union[KVCache, QuantizedKVCache]]: The constructed empty cache.
    """
    num_layers = len(model.layers)
    return [KVCache() for _ in range(num_layers)]
    
def create_cache_dict(cache: List[Union[KVCache, QuantizedKVCache]]) -> Dict[str, mx.array]:
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
    
def save_cache(cache: List[Union[KVCache, QuantizedKVCache]], file: str, metadata: Optional[Dict[str, str]] = None) -> None:
    """Save cache as a safetensors file.

    This function saves a given cache as a safetensors file. The cache is first
    converted to a dictionary of tensors using the `create_cache_dict` function.
    The resulting dictionary is then saved to the specified file using the
    `safetensors.save_file` function. If metadata is provided, it is included in
    the saved file.

    Args:
        cache (List[Union[KVCache, QuantizedKVCache]]): The cache to save.
        file (str): The file path to save the cache to.
        metadata (Optional[Dict[str, str]], optional): Metadata to include in the saved file. Defaults to None.
    """
    cache_dict = create_cache_dict(cache=cache)
    if cache_dict:
        mx.save_safetensors(file=file, arrays=cache_dict, metadata=metadata)
    del cache_dict
    mx.metal.clear_cache()

def load_cache(file: str) -> Tuple[List[Union[KVCache, QuantizedKVCache]], Dict[str, str]]:
    """Load a cache file.

    This function loads a cache file saved as a safetensors file. It returns the loaded cache
    and its metadata.

    Args:
        file (str): The path to the safetensors file containing the cache.

    Returns:
        Tuple[List[Union[KVCache, QuantizedKVCache]], Dict[str, str]]: A tuple containing the loaded cache
            and its metadata.
    """
    cache_dict, metadata = mx.load(file, return_metadata=True)
    cache_list = tree_unflatten(list(cache_dict.items()))
    cache = []
    for i, (key, value) in enumerate(cache_list):
        cache.append(KVCache())
        cache[i].update_and_fetch(key, value)
    mx.eval([c.state for c in cache])
    del cache_dict
    mx.metal.clear_cache()
    return cache, metadata

def split_cache(cache: List[Union[KVCache, QuantizedKVCache]]) -> List[List[KVCache]]:
    """Split the cache into multiple smaller caches. Each smaller cache will contain a subset of the original cache's data.

    Args:
        cache (List[Union[KVCache, QuantizedKVCache]]): The cache to be split.

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
        cache: List[Union[KVCache, QuantizedKVCache]],
        prompt_index: int = 0,
        select_len: Optional[int] = None,
        start_from: int = 0
    ) -> List[Tuple[mx.array, mx.array]]:
    """Get the kv cache tensors of a specific sequence in a cache with multiple prompt cache sequences.

    Args:
        cache (List[Union[KVCache, QuantizedKVCache]]): Cache object.
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
