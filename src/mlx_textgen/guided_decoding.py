import warnings
import dataclasses
from .caching import CacheManager
from .utils import remove_bos_duplicates, get_kv_caches, StepOutput, CacheHistory, EngineOutput
import mlx.nn as nn
import mlx.core as mx
import mlx_lm
from mlx_lm.tokenizer_utils import TokenizerWrapper
from typing import Optional, Union, List, Iterator, Generator, Tuple, Dict, Any
try:
    from outlines.processors.base_logits_processor import OutlinesLogitsProcessor
    from outlines.processors.structured import RegexLogitsProcessor, JSONLogitsProcessor, CFGLogitsProcessor
    from outlines.models.transformers import TransformerTokenizer
    OUTLINE_INSTALLED = True
except:
    warnings.warn('Latest version of outline is not installed. Guided decoding is not enabled.')
    OUTLINE_INSTALLED = False

class MlxLLM:

    def __init__(self, model: nn.Module, tokenizer: TokenizerWrapper, cache_manager: CacheManager) -> None:
        self.model = model
        self.mlx_tokenizer = tokenizer
        self.tokenizer = TransformerTokenizer(self.mlx_tokenizer._tokenizer)
        self.cache_manager = cache_manager

    def generate(
        self,
        prompt: str,
        logits_processor: OutlinesLogitsProcessor,
        max_tokens: int = 512,
        temp: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> str:
        streamer = self.stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temp,
            top_p=top_p,
            logits_processor=logits_processor,
            seed=seed
        )
        text = ''
        for eo in streamer:
            text += eo.text
        return EngineOutput(text=text, token=eo.token, token_ids=eo.token_ids, logprobs=eo.logprobs, prompt_len=eo.prompt_len, finish_reason=eo.finish_reason)

    def stream(
        self,
        prompt: str,
        logits_processor: OutlinesLogitsProcessor,
        max_tokens: int = 512,
        temp: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Iterator[EngineOutput]:
        """Generate text using `mlx_lm`.

        Arguments
        ---------
        prompts
            A prompt or list of prompts.
        generation_parameters
            An instance of `GenerationParameters` that contains the prompt,
            the maximum number of tokens, stop sequences and seed. All the
            arguments to `SequenceGeneratorAdapter`'s `__cal__` method.
        logits_processor
            The logits processor to use when generating text.
        sampling_parameters
            An instance of `SamplingParameters`, a dataclass that contains
            the name of the sampler to use and related parameters as available
            in Outlines.
        Returns
        -------
        The generated text.
        """
        if max_tokens is None:
            max_tokens = int(1e9)

        if not isinstance(prompt, str):
            raise NotImplementedError(
                "The `mlx-lm` library does not support batch inference."
            )

        generate_kwargs = {
            "temp": temp,
            "top_p": top_p,
            "logits_processor": logits_processor,
            "seed": seed
        }
        prompt_tokens = self.mlx_tokenizer.encode(prompt)
        prompt_tokens = remove_bos_duplicates(prompt_tokens, self.mlx_tokenizer.bos_token_id)
        prompt_len = len(prompt_tokens)
        token_array = mx.array(prompt_tokens)
        cache_history, cache_id = self.cache_manager.find_cache(token_ids=prompt_tokens)
        generate_kwargs['cache_history'] = cache_history

        detokenizer = self.mlx_tokenizer.detokenizer
        detokenizer.reset()

        for step, n in zip(
            self.generate_step(prompt=token_array, **generate_kwargs),
            range(max_tokens),
        ):
            if step.token == self.tokenizer.eos_token_id:
                break
            detokenizer.add_token(step.token)
            eo = EngineOutput(text=detokenizer.last_segment, token=step.token, token_ids=step.token_ids, logprobs=step.logprobs, prompt_len=prompt_len, finish_reason=None)
            yield eo

        detokenizer.finalize()
        self.cache_manager.save_cache(step, from_cache_id=cache_id)
        eo = EngineOutput(text=detokenizer.last_segment, token=step.token, token_ids=step.token_ids, logprobs=step.logprobs, prompt_len=prompt_len, finish_reason='stop')
        yield eo

    def generate_step(
            self,
            prompt: mx.array,
            temp: Optional[float],
            top_p: Optional[float],
            logits_processor: OutlinesLogitsProcessor,
            seed: Optional[int] = None,
            cache_history: Union[StepOutput, CacheHistory] = None
        ) -> Generator[StepOutput, None, None]:
            """
            Adapted from
            https://github.com/ml-explore/mlx-examples/blob/4872727/llms/mlx_lm/utils.py#L129

            A generator producing token ids based on the given prompt from the model.

                Args:
                    prompt (mx.array): The input prompt.
                    temp (float): The temperature for sampling, if 0 the argmax is used.
                    Default: ``0``.
                    top_p (float, optional): Nulceus sampling, higher means model considers
                    more less likely words.
                    sampler (str): The sampler string defined by SequenceGeneratorAdapter
                    logits_processor (OutlinesLogitsProcessor): Augment logits before sampling.
            """
            temperature: float = temp or 1.0

            if seed:
                mx.random.seed(seed)

            def sample(logits: mx.array) -> Tuple[mx.array, float]:
                softmax_logits = mx.softmax(logits)

                if temperature == 0.0:
                    token = mx.argmax(logits, axis=-1)
                elif top_p is not None:
                    if top_p > 0 and top_p < 1.0:
                        token = mlx_lm.sample_utils.top_p_sampling(
                            logits, top_p, temperature
                        )
                    else:
                        token = mx.random.categorical(logits * (1 / temperature))
                else:
                    token = mx.random.categorical(logits * (1 / temperature))

                prob = softmax_logits[0, token]
                return token, prob
            
            tokens = prompt.tolist()
            cache, max_prefix = get_kv_caches(model=self.model, promp_tokens=tokens, max_kv_size=None, cache_history=cache_history)

            # kv cache contains processed input IDs, we pass the unprocessed inputs and cache to model()
            unprocessed_input_ids = prompt[max_prefix:]
            generated_ids: List[int] = []

            while True:
                logits = self.model(unprocessed_input_ids[None], cache=cache)
                logits = logits[:, -1, :]

                if logits_processor is not None:
                    # convert to logits_processor 1d expectation, apply, then convert back
                    logits_1d = logits.reshape(-1)
                    logits_1d = logits_processor(generated_ids, logits_1d)
                    logits = logits_1d.reshape(1, -1)

                new_token_single, prob = sample(logits)
                new_token = new_token_single.item()
                output = StepOutput(token=new_token, logprobs=prob, token_ids=generated_ids, cache=cache)
                yield output

                generated_ids.append(new_token)
                unprocessed_input_ids = new_token_single

def get_regex_processor(regex_str: str, llm: MlxLLM) -> RegexLogitsProcessor:
    return RegexLogitsProcessor(regex_str, tokenizer=llm.tokenizer)

def get_json_processor(schema: Union[Dict[str, Any], str], llm: MlxLLM, whitespace_pattern: Optional[str] = None) -> JSONLogitsProcessor:
    return JSONLogitsProcessor(schema=schema, tokenizer=llm.tokenizer, whitespace_pattern=whitespace_pattern)

def get_choice_processor(choices: List[str], llm: MlxLLM) -> RegexLogitsProcessor:
    regex_str = r"(" + r"|".join(choices) + r")"
    return get_regex_processor(regex_str=regex_str, llm=llm)

def get_grammar_processor(cfg_str: str, llm: MlxLLM) -> CFGLogitsProcessor:
    return CFGLogitsProcessor(cfg_str=cfg_str, tokenizer=llm.tokenizer)
