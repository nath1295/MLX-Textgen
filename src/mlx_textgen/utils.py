from mlx_lm.utils import make_kv_caches, load, convert, apply_repetition_penalty
from mlx_lm.models.base import KVCache, RotatingKVCache
from mlx_lm.sample_utils import categorical_sampling, min_p_sampling, top_p_sampling
from mlx_lm.tokenizer_utils import TokenizerWrapper
import mlx.core as mx
import mlx.nn as nn
import time, os
from transformers import PreTrainedTokenizer
from typing import Union, Optional, List, Tuple, Dict, Generator, NamedTuple, Callable, Iterator
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

PACKAGE_NAME=  'mlx_textgen'

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
        logit_bias (dictionary, optional): Additive logit bias.
        prefill_step_size (int): Step size for processing the prompt.
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
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: str,
    max_tokens: int = 100,
    stop: Optional[List[str]] = None,
    return_cache: bool = False,
    verbose: bool = False,
    **kwargs,
) -> Generator[Union[str, Tuple[str, StepOutput]], None, None]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        max_tokens (int): The maximum number of tokens to generate.
        stop (Optional[List[str]], optional): List of words to stop generation. The stop words will be returned. Defaults to None.
        return_cache (bool, optional): Whether to return the last step output.
        verbose (bool, optional): Whether to print prompt processing time and stats. Defaults to False.
        kwargs: The remaining options get passed to :func:`generate_step`.
          See :func:`generate_step` for more details.

    Yields:
        Generator[Union[str, Tuple[str, StepOutput]], None, None]: A generator producing text.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    tokens = tokenizer.encode(prompt)
    # remove duplicated bos tokens
    if len(tokens) > 1:
         if tokens[0] == tokenizer.bos_token_id:
            while tokens[1] == tokenizer.bos_token_id:
                tokens = tokens[:1] + tokens[2:]
    prompt_tokens = mx.array(tokens)
    detokenizer = tokenizer.detokenizer

    detokenizer.reset()
    stop = [] if stop is None else list(filter(lambda x: x != '', stop))
    output_text = ''
    contain_stop = False
    for step_output, n in zip(
        generate_step(prompt_tokens, model, verbose=verbose, **kwargs),
        range(max_tokens),
    ):
        if (step_output.token == tokenizer.eos_token_id) or contain_stop:
            break
        detokenizer.add_token(step_output.token)
        tokens.append(step_output.token)
        last_segment = detokenizer.last_segment
        output_text += last_segment

        if any([x in output_text for x in stop]):
            contain_stop = True

        # Yield the last segment if streaming
        if return_cache:
            yield last_segment, step_output
        else:
            yield last_segment

    detokenizer.finalize()
    if return_cache:
        yield detokenizer.last_segment, step_output
    else:
        yield detokenizer.last_segment

def generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: str,
    max_tokens: int = 100,
    stop: Optional[List[str]] = None,
    return_cache: bool = False,
    verbose: bool = False,
    formatter: Optional[Callable] = None,
    **kwargs,
) -> Union[str, Tuple[str, StepOutput]]:
    """
    Generate a complete response from the model.

    Args:
        model (nn.Module): The language model.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        prompt (str): The string prompt.
        max_tokens (int): The maximum number of tokens. Default: ``100``.
        stop (Optional[List[str]], optional): List of words to stop generation. The stop words will be returned. Defaults to None.
        return_cache (bool, optional): Whether to return the last step output.
        verbose (bool, optional): Whether to print prompt processing time and stats. Defaults to False.
        formatter (Optional[Callable]): A function which takes a token and a
            probability and displays it.
        kwargs: The remaining options get passed to :func:`generate_step`.
            See :func:`generate_step` for more details.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    tokens = tokenizer.encode(prompt)
    # remove duplicated bos tokens
    if len(tokens) > 1:
         if tokens[0] == tokenizer.bos_token_id:
            while tokens[1] == tokenizer.bos_token_id:
                tokens = tokens[:1] + tokens[2:]
    prompt_tokens = mx.array(tokens)
    detokenizer = tokenizer.detokenizer

    detokenizer.reset()
    stop = [] if stop is None else list(filter(lambda x: x != '', stop))
    output_text = ''
    contain_stop = False

    for step_output, n in zip(
        generate_step(prompt_tokens, model, verbose=verbose, **kwargs),
        range(max_tokens),
    ):
        if n == 0:
            tic = time.perf_counter()
        if (step_output.token == tokenizer.eos_token_id) or contain_stop:
            break
        detokenizer.add_token(step_output.token)
        output_text += detokenizer.last_segment
        if any([x in output_text for x in stop]):
            contain_stop = True

        if verbose:
            if formatter:
                # We have to finalize so that the prob corresponds to the last segment
                detokenizer.finalize()
                formatter(detokenizer.last_segment, mx.exp(step_output.logprobs[step_output.token]).item())

    token_count = n + 1
    detokenizer.finalize()

    if verbose:
        gen_time = time.perf_counter() - tic
        print(detokenizer.last_segment, flush=True)
        print("=" * 10)
        if token_count == 0:
            print("No tokens generated for this prompt")
            return
        gen_tps = (token_count - 1) / gen_time
        print(f"Generation: {token_count} tokens, {gen_tps:.3f} tokens-per-sec")
        peak_mem = mx.metal.get_peak_memory() / 2**30
        print(f"Peak memory: {peak_mem:.3f} GB")
    
    return detokenizer.text, step_output if return_cache else detokenizer.text

def find_roots(text: str, stop: List[str], stop_len: List[int]) -> Tuple[str, str]:
    """This function is a helper function for stopping stop words from showing up while doing work streaming in some custom llm classes. Not intended to be used alone.

    Args:
        text (str): Output of the model.
        stop (List[str]): List of stop words.
        stop_len (List[int]): List of the lengths of the stop words.

    Returns:
        Tuple[str, str]: Curated output of the model, potential root of stop words.
    """
    root = ''
    for w in stop:
        if w in text:
            return text.split(w)[0], w
    for i, w in enumerate(stop):
        for j in range(stop_len[i]):
            if text[-(j + 1):]==w[:j+1]:
                root = w[:j+1]
                break
        if root:
            break
    text  = text[:-len(root)] if root else text
    return text, root

def enforce_stop_tokens(text: str, stop: List[str]) -> str:
    """Strip text with the given stop words.

    Args:
        text (str): Text to strip.
        stop (List[str]): List of stop words.

    Returns:
        str: Stripped text.
    """
    stop_pos = list(map(lambda x: text.find(x), stop))
    stop_map = list(zip(stop, stop_pos))
    stop_map = list(filter(lambda x: x[1] != -1, stop_map))
    if len(stop_map) != 0:
        stop_map.sort(key=lambda x: x[1])
        stop_word = stop_map[0][0]
        return text.split(sep=stop_word)[0]
    else:
        return text

def textgen_iterator(text_generator: Iterator[str], stop: List[str]) -> Iterator[str]:
    """Make a text generator stop before spitting out the stop words.

    Args:
        text_generator (Iterator[str]): Text generator to transform.
        stop (List[str]): Stop words.

    Yields:
        Iterator[str]: Text generator with stop words applied.
    """
    text, output, root = '', '', ''
    cont = True
    stop_len = list(map(len, stop))
    for i in text_generator:
        temp = text + root + i
        text, root = find_roots(temp, stop, stop_len)
        if root in stop:
            cont = False
        token = text.removeprefix(output)
        output += token
        if cont:
            yield token
        else:
            yield ''
    if root not in stop:
        yield root
    else:
        yield ''


