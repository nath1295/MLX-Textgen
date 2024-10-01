from mlx_lm.utils import make_kv_caches, load, convert, apply_repetition_penalty
from mlx_lm.models.base import KVCache, RotatingKVCache
from mlx_lm.sample_utils import categorical_sampling, min_p_sampling, top_p_sampling
from mlx_lm.tokenizer_utils import TokenizerWrapper
import mlx.core as mx
import mlx.nn as nn
import time, os
from transformers import PreTrainedTokenizer
from typing import Union, Optional, List, Tuple, Dict, Generator, NamedTuple, Callable, Iterator, Literal
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

PACKAGE_NAME = 'mlx_textgen'

FINISH_REASON = Optional[Literal['stop', 'length']]

def mlx_cache_dir() -> str:
    home_dir = os.path.expanduser('~')
    mlx_cache_dir = os.path.join(home_dir, '.cache', PACKAGE_NAME)
    os.makedirs(mlx_cache_dir, exist_ok=True)
    return mlx_cache_dir

class StepOutput(NamedTuple):
    token: int
    logprobs: mx.array
    token_ids: List[int]
    cache: Optional[List[Union[KVCache, RotatingKVCache]]] = None

class CacheHistory(NamedTuple):
    cache: List[Tuple[mx.array, mx.array]]
    token_ids: List[int]

class GenerateOutput(NamedTuple):
    text: str
    step: StepOutput
    finish_reason: FINISH_REASON
    prompt_len: int

class StopCondition(NamedTuple):
    stop_met: bool
    trim_length: int

def convert_cache_to_history(cache: StepOutput) -> CacheHistory:
    """Helper function to convert the output of "generate_step" into reusable cache history.

    Args:
        cache (StepOutput): Outout of "generate_step".

    Returns:
        CacheHistory: Reusable cache history.
    """
    cache_list = [(c.state[0][..., : c.offset, :], c.state[1][..., : c.offset, :]) for c in cache.cache]
    return CacheHistory(cache=cache_list, token_ids=cache.token_ids)

def save_cache(cache: Union[StepOutput, CacheHistory], filename: str, metadata: Optional[Dict[str, str]] = None) -> None:
    """Saving a prompt cache into a disk.

    Args:
        cache (Union[StepOutput, CacheHistory]): Output of "generate_step" or formatted cache history.
        filename (str): File directory of the prompt cache safetensors file.
        metadata (Optional[Dict[str, str]], optional): String keys and values metadata for the cache file. Defaults to None.
    """
    import orjson
    cache = cache if isinstance(cache, CacheHistory) else convert_cache_to_history(cache=cache)
    metadata = dict() if not isinstance(metadata, dict) else metadata
    metadata['token_ids'] = orjson.dumps(cache.token_ids).decode()
    cache_dict = {}
    for i, c in enumerate(cache.cache):
        cache_dict[f'{i}_key'] = c[0]
        cache_dict[f'{i}_value'] = c[1]
    mx.save_safetensors(file=filename, arrays=cache_dict, metadata=metadata)
    mx.metal.clear_cache()

def load_cache(filename: str) -> Tuple[CacheHistory, Dict[str, str]]:
    """Loading prompt cache from a safetnesors file into ram for generation.

    Args:
        filename (str): File directory of the prompt cache safetensors file. 

    Returns:
        Tuple[CacheHistory, Dict[str, str]]: Reusable cache history and the metadata.
    """
    import orjson
    cache_dict, metadata = mx.load(filename, return_metadata=True)

    # Loading cache
    num_layers = int(len(cache_dict) / 2)
    cache = []
    for i in range(num_layers):
        cache.append((cache_dict[f'{i}_key'], cache_dict[f'{i}_value']))
    token_ids = orjson.loads(metadata.pop('token_ids'))
    ch = CacheHistory(cache=cache, token_ids=token_ids)
    mx.metal.clear_cache()
    return ch, metadata

def remove_bos_duplicates(token_ids: List[int], bos_token_id: Optional[int]) -> List[int]:
    """Remove duplicates of bos tokens if there are multiple of them in the token id list. This usually happen when the prompt was already formatted by the tokenizer.

    Args:
        token_ids (List[int]): List of token ids.
        bos_token_id (Optional[int]): BOS token id.

    Returns:
        List[int]: List of token ids without bos token duplicates.
    """
    if ((len(token_ids) > 1) and (token_ids[0] == bos_token_id)):
        while token_ids[1] == bos_token_id:
            token_ids = token_ids[:1] + token_ids[2:]
    return token_ids

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

def stopping_criteria(
        text: str,
        stop_tuple: List[Tuple[str, int]],
        eos_tuple: Union[Tuple[str, int], None],
    ) -> StopCondition:
    """Get the stopping condition with stop words and eos token.

    Args:
        text (str): Text to be determined to stop or not.
        stop_tuple (List[Tuple[str, int]]): List of tuple with the stop word string and the length of the string. Must be ordered descendingly by length.
        eos_tuple (Union[Tuple[str, int], None]): Eos token string and it's length.

    Returns:
        StopCondition: A status class stating whether the generation should stop and the length of text to trim.
    """
    if eos_tuple is not None and eos_tuple[0] in text:
        return StopCondition(stop_met=True, trim_length=len(text.split(eos_tuple)[-1]) + eos_tuple[1])

    return next(
        (StopCondition(stop_met=True, trim_length=length + len(text.split(stop)[-1]))
        for stop, length in stop_tuple if stop in text), StopCondition(stop_met=False, trim_length=0)
    )

def sequence_overlap(s1: str, s2: str) -> bool:
    """Determine if there is overlapping string between the suffix of the first string and the prefix of the second string.

    Args:
        s1 (str): First string.
        s2 (str): Second string.

    Returns:
        bool: Whether there is overlap.
    """
    max_overlap = min(len(s1), len(s2))
    return any(s1[-i:] == s2[:i] for i in range(1, max_overlap + 1))

def get_kv_caches(
        model: nn.Module, 
        promp_tokens: List[int],
        max_kv_size: Optional[int] = None, 
        cache_history: Optional[Union[CacheHistory, StepOutput]] = None
    ) -> Tuple[List[Union[KVCache, RotatingKVCache]], int]:
    """Helper function to setup the kv cache in "generate_step".

    Args:
        model (nn.Module): The LLM model.
        promp_tokens (List[int]): Prompt tokens ids.
        max_kv_size (Optional[int], optional): Maximum size of the key-value cache. Old entries (except the first 4 tokens) will be overwritten.. Defaults to None.
        cache_history (Optional[Union[CacheHistory, StepOutput]], optional): Reusable prompt cache history or previous generation step output. Defaults to None.

    Returns:
        Tuple[List[Union[KVCache, RotatingKVCache]], int]: List of KV cache for model generation and the number of tokens reused from the cache history.
    """
    cache = make_kv_caches(model=model, max_kv_size=max_kv_size)
    max_prefix = 0

    if cache_history is not None:
        if isinstance(cache_history, StepOutput):
            cache_history = convert_cache_to_history(cache=cache_history)
        if len(cache_history.cache) != len(cache):
            raise ValueError("Wrong number of layers in the cache history")
        cache_size = cache_history.cache[0][0].shape[2]
        if (max_kv_size is None) or (cache_size <= max_kv_size):
            max_prefix = find_max_prefix_num(promp_tokens, cache_history.token_ids)
            # Leave at least one token to evaluate during generation.
            max_prefix = max_prefix - 1 if len(promp_tokens) == max_prefix else max_prefix

            # Set the history in the cache objects and evaluate them to prepare for
            # generation.
            for c, h in zip(cache, cache_history.cache):
                c.update_and_fetch(h[0][:, :, :max_prefix, :], h[1][:, :, :max_prefix, :])
            mx.eval([c.state for c in cache])
    return cache, max_prefix
    
def generate_step(
    prompt: mx.array,
    model: nn.Module,
    temp: float = 0.0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    top_p: float = 1.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    seed: Optional[int] = None,
    logit_bias: Optional[Dict[int, float]] = None,
    prefill_step_size: int = 512,
    verbose: bool = False,
    max_kv_size: Optional[int] = None,
    cache_history: Optional[Union[CacheHistory, StepOutput]] = None
) -> Generator[StepOutput, None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        temp (float): The temperature for sampling, if 0 the argmax is used.
          Default: ``0``.
        repetition_penalty (float, optional): The penalty factor for repeating
          tokens.
        repetition_context_size (int, optional): The number of tokens to
          consider for repetition penalty. Default: ``20``.
        top_p (float, optional): Nulceus sampling, higher means model considers
          more less likely words.
        min_p (float, optional): The minimum value (scaled by the top token's
          probability) that a token probability must have to be considered.
        min_tokens_to_keep (int, optional): Minimum number of tokens that cannot
          be filtered by min_p sampling.
        seed (Optional[int], optional): Random seed for sampling. Defaults to None.
        logit_bias (dictionary, optional): Additive logit bias.
        prefill_step_size (int, optional): Step size for processing the prompt. Defaults to 512.
        verbose (bool, optional): 
        max_kv_size (int, optional): Maximum size of the key-value cache. Old
          entries (except the first 4 tokens) will be overwritten.
        cache_history (Optional[Union[CacheHistory, StepOutput]], optional): Reusable prompt cache history or previous generation step output. Defaults to None.

    Yields:
        Generator[StepOutput, None, None]: A generator producing
          one token and a vector of log probabilities.
    """

    def sample(logits: mx.array) -> Tuple[mx.array, float]:
        if logit_bias:
            indices = mx.array(list(logit_bias.keys()))
            values = mx.array(list(logit_bias.values()))
            logits[:, indices] += values
        logprobs = logits - mx.logsumexp(logits)

        if temp == 0:
            token = mx.argmax(logits, axis=-1)
        else:
            if top_p > 0 and top_p < 1.0:
                token = top_p_sampling(logits, top_p, temp)
            elif min_p != 0.0:
                token = min_p_sampling(logits, min_p, min_tokens_to_keep, temp)
            else:
                token = categorical_sampling(logits, temp)
        return token, logprobs

    if repetition_penalty and (
        repetition_penalty < 0 or not isinstance(repetition_penalty, float)
    ):
        raise ValueError(
            f"repetition_penalty must be a non-negative float, got {repetition_penalty}"
        )
    tokens: List[int] = prompt.tolist()
    token_count = len(tokens)
    y = prompt

    if seed:
        mx.random.seed(seed)

    # Create the KV cache for generation and get the number of tokens being reused.
    cache, max_prefix = get_kv_caches(model=model, promp_tokens=tokens, max_kv_size=max_kv_size, cache_history=cache_history)
    y = y[max_prefix:]

    repetition_context = tokens

    if repetition_context_size:
        repetition_context = repetition_context[-repetition_context_size:]

    def _step(y):
        nonlocal repetition_context
        logits = model(y[None], cache=cache)
        logits = logits[:, -1, :]

        if repetition_penalty:
            logits = apply_repetition_penalty(
                logits, repetition_context, repetition_penalty
            )
            y, logprobs = sample(logits)
            repetition_context.append(y.item())
        else:
            y, logprobs = sample(logits)

        if repetition_context_size:
            if len(repetition_context) > repetition_context_size:
                repetition_context = repetition_context[-repetition_context_size:]
        return y, logprobs.squeeze(0)

    # Getting preprocessing batches
    num_batches = y.shape[0] // prefill_step_size
    if num_batches != (y.size / prefill_step_size):
        num_batches += 1
    batches = [(i * prefill_step_size, min((i + 1) * prefill_step_size, y.size)) for i in range(num_batches)]
    num_tokens = y.size

    # Prompt preprocessing
    if verbose:
        from tqdm import tqdm
        batches = tqdm(batches)
    pp_start = time.perf_counter()
    for b in batches:
        if verbose:
            batches.set_description(f'Processing prompt ({b[1]}/{y.size})')
        if (b[1] - b[0]) >= prefill_step_size:
            model(y[b[0]:b[1]][None], cache=cache)
            mx.eval([c.state for c in cache])
            mx.metal.clear_cache() # Clearing mlx cache, otherwise it grows very quick with longer prompts.
        else:
            y = y[b[0]:b[1]]
            y, logprobs = _step(y)
    pp_end = time.perf_counter() - pp_start
    if verbose:
        print(f'Prompt preprocessing time for {num_tokens} tokens: {pp_end:.4}s ({num_tokens/pp_end:.4f} tok/sec)')

    mx.async_eval(y)
    while True:
        next_y, next_logprobs = _step(y)
        mx.async_eval(next_y)
        token = y.item()
        tokens.append(token)
        token_count += 1
        if max_kv_size is not None:
            # Trim off token ids if max_kv_size is set.
            if token_count >= max_kv_size:
                token_count -= 1
                tokens = tokens[:4] + tokens[5:]
        yield StepOutput(token=token, logprobs=logprobs, token_ids=tokens, cache=cache)
        y, logprobs = next_y, next_logprobs

def stream_generate(
        model: nn.Module, 
        tokenizer: TokenizerWrapper, 
        prompt: str,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        cache_history: Union[StepOutput, CacheHistory] = None,
        verbose: bool = False,
        **kwargs
    ) -> Iterator[GenerateOutput]:
    # prepare prompt
    prompt_tokens = tokenizer.encode(prompt)
    prompt_tokens = remove_bos_duplicates(token_ids=prompt_tokens, bos_token_id=tokenizer.bos_token_id)

    # prepare for generation
    detokenizer = tokenizer._detokenizer
    detokenizer.reset()
    finish_reason = None
    stop = [] if stop is None else list(set(stop))
    stop = list(filter(lambda x: x != '', stop)) # remove empty strings
    stop_tuple = [(s, len(s)) for s in stop]
    stop_tuple.sort(key=lambda x: x[1], reverse=True)
    eos_token = tokenizer.eos_token
    eos_tuple = [eos_token, len(eos_token)] if eos_token else None
    stop_suffix = None
    prompt_len = len(prompt_tokens)
    text = ''
    for step, n in zip(
        generate_step(prompt=mx.array(prompt_tokens), model=model, cache_history=cache_history, verbose=verbose, **kwargs),
        range(max_tokens)
    ):
        if n + 1 == max_tokens:
            finish_reason = 'length'
        detokenizer.add_token(step.token)
        text += detokenizer.last_segment
        sc = stopping_criteria(text=text, stop_tuple=stop_tuple, eos_tuple=eos_tuple)
        if sc.stop_met:
            if sc.trim_length:
                stop_suffix = text[-sc.trim_length:]
                finish_reason = 'stop'
            break
        if any(
            (sequence_overlap(text, s) for s in stop)
        ):
            continue
        new_text = text
        text = ''
        yield GenerateOutput(text=new_text, step=step, finish_reason=finish_reason, prompt_len=prompt_len)

    detokenizer.finalize()
    text += detokenizer.last_segment
    if text:
        if stop_suffix is not None:
            text = text[: -len(stop_suffix)]
        yield GenerateOutput(text=text, step=step, finish_reason=finish_reason, prompt_len=prompt_len)

def generate(
        model: nn.Module, 
        tokenizer: TokenizerWrapper, 
        prompt: str,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
        cache_history: Union[StepOutput, CacheHistory] = None,
        verbose: bool = False,
        **kwargs
    ):
    # prepare prompt
    prompt_tokens = tokenizer.encode(prompt)
    prompt_tokens = remove_bos_duplicates(token_ids=prompt_tokens, bos_token_id=tokenizer.bos_token_id)

    # prepare for generation
    detokenizer = tokenizer._detokenizer
    detokenizer.reset()
    finish_reason = None
    stop = [] if stop is None else list(set(stop))
    stop = list(filter(lambda x: x != '', stop)) # remove empty strings
    stop_tuple = [(s, len(s)) for s in stop]
    stop_tuple.sort(key=lambda x: x[1], reverse=True)
    eos_token = tokenizer.eos_token
    eos_tuple = [eos_token, len(eos_token)] if eos_token else None
    stop_suffix = None
    prompt_len = len(prompt_tokens)
    text = ''
    for step, n in zip(
        generate_step(prompt=mx.array(prompt_tokens), model=model, cache_history=cache_history, verbose=verbose, **kwargs),
        range(max_tokens)
    ):
        if n + 1 == max_tokens:
            finish_reason = 'length'
        detokenizer.add_token(step.token)
        text += detokenizer.last_segment
        sc = stopping_criteria(text=text, stop_tuple=stop_tuple, eos_tuple=eos_tuple)
        if sc.stop_met:
            if sc.trim_length:
                stop_suffix = text[-sc.trim_length:]
                finish_reason = 'stop'
            break

    detokenizer.finalize()
    text += detokenizer.last_segment
    if stop_suffix is not None:
        text = text[: -len(stop_suffix)]
    return GenerateOutput(text=text, step=step, finish_reason=finish_reason, prompt_len=prompt_len)
