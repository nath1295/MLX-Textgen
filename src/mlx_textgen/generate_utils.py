from .tokenizer_utils import TokenizerWrapper
import mlx.core as mx
import mlx.nn as nn
import time
from .cache_utils import CacheHistory, make_empty_cache
from .sampling_utils import categorical_sampling, min_p_sampling, top_p_sampling
from typing import Optional, Literal, List, NamedTuple, Tuple, Generator, Union, Dict, Any
from logging import Logger
import warnings
try:
    from outlines.processors.base_logits_processor import OutlinesLogitsProcessor
    from outlines.processors.structured import RegexLogitsProcessor, JSONLogitsProcessor, CFGLogitsProcessor
    from outlines.models.transformers import TransformerTokenizer
    OUTLINE_INSTALLED = True
except:
    warnings.warn('Latest version of outline is not installed. Guided decoding is not enabled.')
    OUTLINE_INSTALLED = False


FINISH_REASON = Optional[Literal['stop', 'length']]

class StopCondition(NamedTuple):
    stop_met: bool
    trim_length: int
    stop_text: Optional[str] = None

class GenerationOutput(NamedTuple):
    text: Optional[str]
    token: Optional[int]
    token_ids: mx.array
    logprobs: mx.array
    finish_reason: FINISH_REASON
    stop_text: Optional[str] = None

def stopping_criteria(
        text: str,
        stop_tuple: List[Tuple[str, int]]
    ) -> StopCondition:
    """Get the stopping condition with stop words and eos token.

    Args:
        text (str): Text to be determined to stop or not.
        stop_tuple (List[Tuple[str, int]]): List of tuple with the stop word string and the length of the string. Must be ordered descendingly by length.

    Returns:
        StopCondition: A status class stating whether the generation should stop and the length of text to trim.
    """
    return next(
        (StopCondition(stop_met=True, trim_length=len(text) - len(text.split(stop)[0]), stop_text=stop)
        for stop, length in stop_tuple if stop in text), StopCondition(stop_met=False, trim_length=0, stop_text=None)
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

def get_batches(size: int, batch_size: int) -> List[Tuple[int, int]]:
    """Generate a list of batch ranges for a given size and batch size.

    Args:
        size (int): The total size to be divided into batches.
        batch_size (int): The size of each batch.

    Returns:
        List[Tuple[int, int]]: A list of tuples, where each tuple represents the start and end indices of a batch.
    """
    num_batches = size // batch_size if (size // batch_size) == (size / batch_size) else (size // batch_size) + 1
    batches = [(b * batch_size, min(size, (b + 1) * batch_size)) for b in range(num_batches)]
    return batches

def apply_repetition_penalty(logits: mx.array, tokens: mx.array, penalty: float) -> mx.array:
    """
    Apply repetition penalty to specific logits based on the given context.

    Paper: https://arxiv.org/abs/1909.05858

    Args:
        logits (mx.array): The logits produced by the language model.
        tokens (mx.array): A list of N previous tokens.
        penalty (float): The repetition penalty factor to be applied.

    Returns:
        logits (mx.array): Logits with repetition penalty applied to generated tokens.
    """
    if len(tokens) > 0:
        selected_logits = logits[:, tokens]
        selected_logits = mx.where(
            selected_logits < 0, selected_logits * penalty, selected_logits / penalty
        )
        logits[:, tokens] = selected_logits
    return logits

def generate_step(
        model: nn.Module,
        prompt_ids: mx.array,
        temperature: float = 0.0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = 20,
        cache_history: Optional[CacheHistory] = None,
        prefill_batch_size: int = 512,
        logit_bias: Optional[Dict[int, float]] = None,
        logits_processor: Optional[OutlinesLogitsProcessor] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
        logger: Optional[Logger] = None
    ) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    Generate text tokens step by step using a language model.

    This function processes the prompt, applies repetition penalty, and samples the next token based on the given temperature, top_p, and min_p parameters.
    It also supports logit bias and logits processor for custom logit adjustments.

    Args:
        model (nn.Module): The language model to use for generating tokens.
        prompt_ids (mx.array): The initial token ids for the prompt.
        temperature (float): The temperature parameter for sampling.
        top_p (float): The top_p parameter for nucleus sampling.
        min_p (float): The min_p parameter for min_p sampling.
        min_tokens_to_keep (int): The minimum number of tokens to keep for min_p sampling.
        repetition_penalty (Optional[float]): The repetition penalty factor to apply.
        repetition_context_size (Optional[int]): The context size for repetition penalty.
        cache_history (Optional[CacheHistory]): The cache history to use for caching.
        prefill_batch_size (int): The batch size for prefilling.
        logit_bias (Optional[Dict[int, float]]): The logit bias to apply.
        logits_processor (Optional[OutlinesLogitsProcessor]): The logits processor to use for guided decoding.
        seed (Optional[int]): The random seed for reproducibility.
        verbose (bool): Whether to print progress information.

    Yields:
        Tuple[mx.array, mx.array]: The next token ids and their log probabilities.
    """

    # Create or validate the cache history
    num_prompts = prompt_ids.shape[0]
    prompt_size = prompt_ids.shape[1]
    cache_size = 0
    if cache_history is None:
        cache_history = CacheHistory(cache=make_empty_cache(model), token_ids=[[]] * num_prompts)
    elif cache_history.cache[0].offset == 0:
        pass
    elif cache_history.cache[0].state[0].shape[0] != num_prompts:
        raise ValueError('Cache history number of slots does not match with number of prompts.')
    elif cache_history.cache[0].state[0].shape[2] >= prompt_size:
        raise ValueError('Cache history size larger than prompt size.')
    else:
        cache_size = cache_history.cache[0].state[0].shape[2]
    
    # Trimming prompt ids with cache
    unprocessed_ids = prompt_ids[:, cache_size:]

    # Preparing resources
    num_gen_tokens: int = 0
    rng = mx.arange(num_prompts).reshape((-1, 1))
    if seed is not None:
        mx.random.seed(seed)

    # Function for processing the next token
    if logit_bias:
        indices = mx.array(list(logit_bias.keys()))
        values = mx.array(list(logit_bias.values()))

        def logit_bias_processor(logits):
            logits[:, indices] += values
            return logits
        

    def sample(logits: mx.array) -> Tuple[mx.array, mx.array]:
        logprobs = logits - mx.logsumexp(logits)
        if temperature == 0:
            y = mx.argmax(logits, axis=1).reshape(-1, 1)
        elif top_p > 0 and top_p < 1.0:
            y = top_p_sampling(logits=logits, top_p=top_p, temperature=temperature, rng=rng)
        elif min_p != 0.0:
            y = min_p_sampling(logits=logits, min_p=min_p, rng=rng, min_tokens_to_keep=min_tokens_to_keep, temperature=temperature)
        else:
            y = categorical_sampling(logits=logits, temperature=temperature)
        return y, logprobs

    def _step(y: mx.array) -> Tuple[List[int], mx.array]:
        logits = model(y, cache=cache_history.cache)
        logits = logits[:, -1, :]

        if repetition_penalty:
            tks = prompt_ids[:, -repetition_context_size:] if repetition_context_size is not None else prompt_ids
            logits = apply_repetition_penalty(logits=logits, tokens=tks, penalty=repetition_penalty)

        if logit_bias:
            logits = logit_bias_processor(logits)

        if logits_processor:
            input_ids = prompt_ids[:, -num_gen_tokens:] if num_gen_tokens != 0 else [[]] * num_prompts
            logits = logits_processor(input_ids=input_ids, logits=logits)

        y, logprobs = sample(logits)
        return y, logprobs

    # Prompt processing
    batches = get_batches(unprocessed_ids.shape[1], batch_size=prefill_batch_size)
    if verbose:
        from tqdm import tqdm
        batches = tqdm(batches)
    pp_start = time.perf_counter()
    for b1, b2 in batches:
        if verbose:
            batches.set_description(f'Processing prompt ({b2}/{unprocessed_ids.shape[1]})')

        if b2 != unprocessed_ids.shape[1]: # proccesing without sampling and yielding next token
            model(unprocessed_ids[:, b1:b2], cache=cache_history.cache)
            mx.eval([c.state for c in cache_history.cache])
            mx.metal.clear_cache()

        else: # proccesing with sampling and ygetting next tokens with the last batch
            y, logprobs = _step(unprocessed_ids[:, b1:b2])
            prompt_ids = mx.concat([prompt_ids, y], axis=1)
            num_gen_tokens += 1
    pp_end = time.perf_counter() - pp_start
    if verbose:
        num_tokens = unprocessed_ids.shape[1] * num_prompts
        if logger:
            logger.info(f'Prompt preprocessing time for {num_tokens} tokens: {pp_end:.4}s ({num_tokens/pp_end:.4f} tok/sec)')
        else:
            print(f'Prompt preprocessing time for {num_tokens} tokens: {pp_end:.4}s ({num_tokens/pp_end:.4f} tok/sec)')
    mx.async_eval(y)

    # Generating tokens
    while True:
        next_y, next_logprobs = _step(y)
        mx.async_eval(next_y)
        prompt_ids = mx.concat([prompt_ids, next_y], axis=1)
        num_gen_tokens += 1
        yield y, logprobs
        y, logprobs = next_y, next_logprobs
    
def stream_generate(
        model: nn.Module,
        tokenizer: TokenizerWrapper,
        prompt: Union[str, List[str]],
        max_tokens: int = 512,
        stop: Optional[List[str]]  = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = 20,
        cache_history: Optional[CacheHistory] = None,
        prefill_batch_size: int = 512,
        logit_bias: Optional[Dict[int, float]] = None,
        logits_processor: Optional[OutlinesLogitsProcessor] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
        logger: Optional[Logger] = None
    ) -> Generator[List[GenerationOutput], None, None]:
    """
    Stream generate text tokens using a language model.

    This function tokenizes the prompt, processes the tokens, and generates text step by step.
    It supports various sampling methods, repetition penalty, and logit adjustments.

    Args:
        model (nn.Module): The language model to use for generating tokens.
        tokenizer (TokenizerWrapper): The tokenizer to use for tokenizing the prompt.
        prompt (Union[str, List[str]]): The initial prompt to generate text from.
        max_tokens (int): The maximum number of tokens to generate.
        stop (Optional[List[str]]): A list of stop words or tokens to stop generation.
        temperature (float): The temperature parameter for sampling.
        top_p (float): The top_p parameter for nucleus sampling.
        min_p (float): The min_p parameter for min_p sampling.
        min_tokens_to_keep (int): The minimum number of tokens to keep for min_p sampling.
        repetition_penalty (Optional[float]): The repetition penalty factor to apply.
        repetition_context_size (Optional[int]): The context size for repetition penalty.
        cache_history (Optional[CacheHistory]): The cache history to use for caching.
        prefill_batch_size (int): The batch size for prefilling.
        logit_bias (Optional[Dict[int, float]]): The logit bias to apply.
        logits_processor (Optional[OutlinesLogitsProcessor]): The logits processor to use for guided decoding.
        seed (Optional[int]): The random seed for reproducibility.
        verbose (bool): Whether to print progress information.

    Yields:
        List[GenerationOutput]: A list of generation outputs, each containing the generated text, token, token ids, log probabilities, finish reason, and stop text.
    """
    # Preparing prompts
    if isinstance(prompt, str):
        prompt = [prompt]
    bos_token_id = tokenizer.bos_token_id
    pad_token_id = bos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    prompt_ids = [remove_bos_duplicates(token_ids=tokenizer.encode(p), bos_token_id=bos_token_id) for p in prompt]
    prompt_lens = [len(p) for p in prompt_ids]
    max_len = max(prompt_lens)
    prompt_ids = [[pad_token_id] * (max_len - p[1]) + p [0] for p in zip(prompt_ids, prompt_lens)]
    prompt_ids = mx.array(prompt_ids)

    # Preparing stop sequences
    eos_token = tokenizer.eos_token
    stop = [eos_token] if (stop is None) or (logits_processor is not None) else [eos_token] + stop # Ignore stop words if guided decoding is used.
    stop = list(set(stop))
    stop_len = [len(s) for s in stop]
    stop_tuple = list(zip(stop, stop_len))
    stop_tuple.sort(key=lambda x: x[1], reverse=True)

    # Generation
    def process_stop_condition(text: str, 
                               stop: List[str], 
                               stop_condition: StopCondition,
                               is_stop: bool,
                               finish_reason: FINISH_REASON,
                               is_last: bool) -> Tuple[str, str, FINISH_REASON, Optional[str]]:
        if is_stop:
            return '', '', None, None
        elif stop_condition.stop_met:
            return text[:-stop_condition.trim_length], '', 'stop', stop_condition.stop_text
        elif any(sequence_overlap(text, s) for s in stop) and (not is_last):
            return '', text, finish_reason, stop_condition.stop_text
        else:
            return text, '', finish_reason, stop_condition.stop_text


    texts = [''] * prompt_ids.shape[0]
    is_stopped = [False] * prompt_ids.shape[0]
    detokenizer = tokenizer._detokenizer
    detokenizer.reset()
    try:
        if verbose:
            start = None
        for (tokens, logprobs), n in zip(
            generate_step(
                model=model,
                prompt_ids=prompt_ids,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                min_tokens_to_keep=min_tokens_to_keep,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
                cache_history=cache_history,
                prefill_batch_size=prefill_batch_size,
                logit_bias=logit_bias,
                logits_processor=logits_processor,
                seed=seed,
                verbose=verbose,
                logger=logger
            ),
            range(max_tokens)
        ):
            if verbose:
                start = time.perf_counter() if n == 0 else start
            finish_reason = None if (n + 1) != max_tokens else 'length'
            new_tokens = tokens.tolist()
            detokenizer.add_tokens(new_tokens)
            # finalize for last token
            if (n + 1) == max_tokens:
                detokenizer.finalize()
            new_token_str = detokenizer.last_segments
            prompt_ids = mx.concat([prompt_ids, tokens], axis=1)
            texts = [t[0] + t[1] for t in zip(texts, new_token_str)]
            stop_conditions = [stopping_criteria(t, stop_tuple=stop_tuple) for t in texts]
            outputs = [process_stop_condition(t, stop, sc, s, finish_reason, finish_reason=='length') for t, sc, s in zip(texts, stop_conditions, is_stopped)]
            out_texts, texts, finish_reasons, stop_texts = [list(ls) for ls in zip(*outputs)]
            is_stopped = [s or (fr == 'stop') for s, fr in zip(is_stopped, finish_reasons)]
            gen_outputs = zip(out_texts, new_tokens, prompt_ids, logprobs, finish_reasons, prompt_lens, stop_texts)
            gen_outputs = [GenerationOutput(text=go[0], 
                        token=go[1][0], 
                        token_ids=go[2][-(go[5] + n +1):], 
                        logprobs=go[3][None], 
                        finish_reason=go[4],
                        stop_text=go[6]) for go in gen_outputs]
            yield gen_outputs
            if all(is_stopped):
                break
        
    finally:
        if verbose:
            end = time.perf_counter() - start
            num_tokens = (n + 1) * len(prompt_ids)
            if logger:
                logger.info(f'Number of tokens generated: {num_tokens}; Generation time: {end}s ({(num_tokens / end):.4f} tok/sec)')
            else:
                print(f'Number of tokens generated: {num_tokens}; Generation time: {end}s ({(num_tokens / end):.4f} tok/sec)')
        mx.metal.clear_cache()
    
def generate(
        model: nn.Module,
        tokenizer: TokenizerWrapper,
        prompt: Union[str, List[str]],
        max_tokens: int = 512,
        stop: Optional[List[str]]  = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = 20,
        cache_history: Optional[CacheHistory] = None,
        prefill_batch_size: int = 512,
        logit_bias: Optional[Dict[int, float]] = None,
        logits_processor: Optional[OutlinesLogitsProcessor] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
        logger: Optional[Logger] = None
    ) -> List[GenerationOutput]:
    """
    Generate text tokens using a language model.

    This function tokenizes the prompt, processes the tokens, and generates text step by step.
    It supports various sampling methods, repetition penalty, and logit adjustments.

    Args:
        model (nn.Module): The language model to use for generating tokens.
        tokenizer (TokenizerWrapper): The tokenizer to use for tokenizing the prompt.
        prompt (Union[str, List[str]]): The initial prompt to generate text from.
        max_tokens (int): The maximum number of tokens to generate.
        stop (Optional[List[str]]): A list of stop words or tokens to stop generation.
        temperature (float): The temperature parameter for sampling.
        top_p (float): The top_p parameter for nucleus sampling.
        min_p (float): The min_p parameter for min_p sampling.
        min_tokens_to_keep (int): The minimum number of tokens to keep for min_p sampling.
        repetition_penalty (Optional[float]): The repetition penalty factor to apply.
        repetition_context_size (Optional[int]): The context size for repetition penalty.
        cache_history (Optional[CacheHistory]): The cache history to use for caching.
        prefill_batch_size (int): The batch size for prefilling.
        logit_bias (Optional[Dict[int, float]]): The logit bias to apply.
        logits_processor (Optional[OutlinesLogitsProcessor]): The logits processor to use for guided decoding.
        seed (Optional[int]): The random seed for reproducibility.
        verbose (bool): Whether to print progress information.

    Returns:
        List[GenerationOutput]: A list of generation outputs, each containing the generated text, token, token ids, log probabilities, finish reason, and stop text.
    """
    # Preparing prompts
    if isinstance(prompt, str):
        prompt = [prompt]
    bos_token_id = tokenizer.bos_token_id
    pad_token_id = bos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    prompt_ids = [remove_bos_duplicates(token_ids=tokenizer.encode(p), bos_token_id=bos_token_id) for p in prompt]
    prompt_lens = [len(p) for p in prompt_ids]
    max_len = max(prompt_lens)
    prompt_ids = [[pad_token_id] * (max_len - p[1]) + p [0] for p in zip(prompt_ids, prompt_lens)]
    prompt_ids = mx.array(prompt_ids)

    # Preparing stop sequences
    eos_token = tokenizer.eos_token
    stop = [eos_token] if (stop is None) or (logits_processor is not None) else [eos_token] + stop # Ignore stop words if guided decoding is used.
    stop = list(set(stop))
    stop_len = [len(s) for s in stop]
    stop_tuple = list(zip(stop, stop_len))
    stop_tuple.sort(key=lambda x: x[1], reverse=True)

    # Generation
    texts = [''] * prompt_ids.shape[0]
    is_stopped = [False] * prompt_ids.shape[0]
    stop_texts = [None] * prompt_ids.shape[0]
    detokenizer = tokenizer._detokenizer
    detokenizer.reset()
    try:
        if verbose:
            start = None
        for (tokens, logprobs), n in zip(
            generate_step(
                model=model,
                prompt_ids=prompt_ids,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                min_tokens_to_keep=min_tokens_to_keep,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
                cache_history=cache_history,
                prefill_batch_size=prefill_batch_size,
                logit_bias=logit_bias,
                logits_processor=logits_processor,
                seed=seed,
                verbose=verbose
            ),
            range(max_tokens)
        ):
            if verbose:
                start = time.perf_counter() if n == 0 else start
            new_tokens = tokens.tolist()
            detokenizer.add_tokens(new_tokens)
            # finalize for last token
            if (n + 1) == max_tokens:
                detokenizer.finalize()
            new_token_str = detokenizer.last_segments
            prompt_ids = mx.concat([prompt_ids, tokens], axis=1)
            texts = [t[0] + t[1] if not t[2] else t[0] for t in zip(texts, new_token_str, is_stopped)]
            stop_conditions = [stopping_criteria(t, stop_tuple=stop_tuple) for t in texts]
            stop_texts = [sc.stop_text if sc.stop_met and (not s) else st for s, sc, st in zip(is_stopped, stop_conditions, stop_texts)]
            is_stopped = [s or sc.stop_met for s, sc in zip(is_stopped, stop_conditions)]
            if all(is_stopped):
                break
        if verbose:
            end = time.perf_counter() - start
            num_tokens = (n + 1) * len(prompt_ids)
            if logger:
                logger.info(f'Number of tokens generated: {num_tokens}; Generation time: {end}s ({(num_tokens / end):.4f} tok/sec)')
            else:
                print(f'Number of tokens generated: {num_tokens}; Generation time: {end}s ({(num_tokens / end):.4f} tok/sec)')

        texts = [t[:-sc.trim_length] if s else t for t, sc, s in zip(texts, stop_conditions, is_stopped)]
        finish_reasons = ['stop' if s else 'length' for s in is_stopped]
        gen_outputs = zip(texts, new_tokens, prompt_ids, logprobs, finish_reasons, prompt_lens, stop_texts)
        gen_outputs = [GenerationOutput(text=go[0], token=go[1][0], token_ids=go[2][-(go[5] + n + 1):], logprobs=go[3][None], finish_reason=go[4], stop_text=go[6]) for go in gen_outputs]
        return gen_outputs
        
    finally:
        mx.metal.clear_cache()


# Guided decoding logits processors
def get_regex_processor(regex_str: str, tokenizer: TransformerTokenizer) -> RegexLogitsProcessor:
    return RegexLogitsProcessor(regex_str, tokenizer=tokenizer)

def get_json_processor(schema: Union[Dict[str, Any], str], tokenizer: TransformerTokenizer, whitespace_pattern: Optional[str] = None) -> JSONLogitsProcessor:
    return JSONLogitsProcessor(schema=schema, tokenizer=tokenizer, whitespace_pattern=whitespace_pattern)

def get_choice_processor(choices: List[str], tokenizer: TransformerTokenizer) -> RegexLogitsProcessor:
    regex_str = r"(" + r"|".join(choices) + r")"
    return get_regex_processor(regex_str=regex_str, tokenizer=tokenizer)

def get_grammar_processor(cfg_str: str, tokenizer: TransformerTokenizer) -> CFGLogitsProcessor:
    return CFGLogitsProcessor(cfg_str=cfg_str, tokenizer=tokenizer)
