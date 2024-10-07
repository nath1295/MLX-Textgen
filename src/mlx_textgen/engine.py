from mlx_lm.utils import load, get_model_path, convert, load_config, load_model
from mlx_lm.tokenizer_utils import load_tokenizer
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx.nn import Module
import mlx.core as mx
from .utils import mlx_cache_dir, generate, stream_generate, remove_bos_duplicates, EngineOutput
from .guided_decoding import MlxLLM, get_choice_processor, get_json_processor, get_regex_processor, get_grammar_processor, OUTLINE_INSTALLED
from .caching import CacheManager
from pathlib import Path
from logging import Logger
import os
from typing import NamedTuple, Optional, List, Dict, Union, Literal, Any, Tuple, Iterator

class ModelConfig(NamedTuple):
    model_id_or_path: str
    tokenizer_id_or_path: Optional[str] = None
    adapter_path: Optional[str] = None
    quant: Literal['fp16', 'q8', 'q4', 'q2'] = 'fp16'
    revision: Optional[str] = None
    model_name: Optional[str] = None
    model_config: Optional[Dict[str, Any]] = None
    tokenizer_config: Optional[Dict[str, Any]] = None

def get_model_name(model_id_or_path: str) -> str:
    """Getting a clean name for the model.

    Args:
        model_id_or_path (str): HuggingFace repository name or the local path of the model.

    Returns:
        str: A clean name for the model.
    """
    base_name = model_id_or_path.split('/')[-1]
    base_name = base_name.split('\\')[-1]
    base_name = base_name.lower().replace('_', '-').strip()
    return base_name

def make_model_exist(
        model_id_or_path: str,
        quant: Literal['fp16', 'q8', 'q4', 'q2'] = 'fp16',
        revision: Optional[str] = None,
        adapter_path: Optional[str] = None
    ) -> Path:
    """Check if the the model exists locally, if not, it will attempt to download the model and convert it to the appropriate quant.

    Args:
        model_id_or_path (str): HuggingFace repository name or the local path of the model.
        quant (Literal[&#39;fp16&#39;, &#39;q8&#39;, &#39;q4&#39;, &#39;q2&#39;], optional): Quantisation of the model. Defaults to 'fp16'.
        revision (Optional[str], optional): Revision of the repository if a HuggingFace repository name is given. Defaults to None.
        adapter_path (Optional[str], optional): Check the existence of the adapter file if given. Defaults to None.

    Returns:
        Path: Path to the local model to serve.
    """
    model_base_name = get_model_name(model_id_or_path=model_id_or_path)
    if quant != 'fp16':
        model_base_name += f'-{quant}'
    mlx_path = os.path.join(mlx_cache_dir(), model_base_name)
    
    if os.path.exists(mlx_path):
        if adapter_path is not None:
            if not os.path.exists(adapter_path):
                raise FileNotFoundError(f'Adapter not found in "{adapter_path}".')
        return Path(mlx_path)
    else:
        hf_path = get_model_path(path_or_hf_repo=model_id_or_path, revision=revision)
        if quant == 'fp16':
            mlx_path = hf_path
        else:
            config = load_config(model_path=hf_path)
            quantisation = config.get('quantization', {}).get('bits')
            quant_target = int(quant.removeprefix('q'))
            if quant_target == quantisation:
                mlx_path = hf_path
            else:
                model_base_name = get_model_name(model_id_or_path=model_id_or_path)
                mlx_path = os.path.join(mlx_cache_dir(), model_base_name + f'-q{quant_target}')
                if not os.path.exists(mlx_path):
                    convert(hf_path=hf_path, mlx_path=mlx_path, quantize=True, q_bits=int(quant[-1]))
        if adapter_path is not None:
            if not os.path.exists(adapter_path):
                raise FileNotFoundError(f'Adapter not found in "{adapter_path}".')
        return Path(mlx_path)

def get_model_and_tokenizer(
        model_id_or_path: str,
        tokenizer_id_or_path: Optional[str] = None,
        quant: Literal['fp16', 'q8', 'q4', 'q2'] = 'fp16',
        revision: Optional[str] = None,
        adapter_path: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
        tokenizer_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[Module, TokenizerWrapper]:
    """Load the llm model and it's tokenizer.

    Args:
        model_id_or_path (str): HuggingFace repository name or the local path of the model.
        tokenizer_id_or_path (Optional[str], optional): HuggingFace repository name or the local path of the model. If None is given, `model_id_or_path` will be used. Defaults to None.
        quant (Literal[&#39;fp16&#39;, &#39;q8&#39;, &#39;q4&#39;, &#39;q2&#39;], optional): Quantisation of the model. Defaults to 'fp16'.
        revision (Optional[str], optional): Revision of the repository if a HuggingFace repository name is given. Defaults to None.
        adapter_path (Optional[str], optional): The adapter file for the model. Defaults to None.
        model_config (Optional[Dict[str, Any]], optional): Extra arguments for loading the model. Defaults to None.
        tokenizer_config (Optional[Dict[str, Any]], optional): Extra arguments for loading the tokenizer. Defaults to None.

    Returns:
        Tuple[Module, TokenizerWrapper]: The model and the tokenizer.
    """
    mlx_path = make_model_exist(model_id_or_path=model_id_or_path, quant=quant, revision=revision, adapter_path=adapter_path)
    tokenizer_config = {} if tokenizer_config is None else tokenizer_config
    model_config = {} if model_config is None else model_config
    model = load_model(model_path=mlx_path, model_config=model_config)
    tokenizer = load_tokenizer(model_path=mlx_path if tokenizer_id_or_path is None else Path(tokenizer_id_or_path), tokenizer_config_extra=tokenizer_config)
    return model, tokenizer

class ModelEngine:

    def __init__(self, 
            models: Union[ModelConfig, List[ModelConfig]],
            prefill_step_size: int = 512,
            token_threshold: int = 50, 
            max_keep: int = 10, 
            logger: Optional[Logger] = None
        ) -> None:
        """Initialising the engine.

        Args:
            models (Union[ModelConfig, List[ModelConfig]]): Model configurations or list of model configurations to serve.
            prefill_step_size (int, optional): Batch size for prompt preprocessing. Defaults to 512.
            token_threshold (int, optional): Minimum number of tokens to be considered to save as a prompt cache history. Defaults to 50.
            max_keep (int, optional): Maximum number of cache history files to keep. Defaults to 10.
            logger (Optional[Logger], optional): If a logger is provided, messages of adding and deleting caches will be added.
        """
        self.model_configs = [models] if isinstance(models, ModelConfig) else models
        self.prefill_step_size = prefill_step_size
        self.token_threshold = token_threshold
        self.max_keep = max_keep
        self.logger = logger
        self.models: Dict[str, ModelConfig] = dict()
        self.model: Optional[Module] = None
        self.tokenizer: Optional[TokenizerWrapper] = None
        self.current_model: Optional[str] = None
        self.cache_manager: Optional[CacheManager] = None
        self.outline_model: Optional[MlxLLM] = None
        self.cache_dir = os.path.join(mlx_cache_dir(), 'prompt_cache')
        for model_config in self.model_configs:
            self._prepare_model(model_config=model_config)

    def _prepare_model(self, model_config: ModelConfig) -> None:
        """Making sure the given model is available locally.

        Args:
            model_config (ModelConfig): Model configuaration of the model to check.
        """
        model_path = model_config.model_id_or_path
        adapter_path = model_config.adapter_path
        quant = model_config.quant
        model_name = model_config.model_name
        if model_name is None:
            model_name = get_model_name(model_path)
            if quant != 'fp16':
                model_name += f'-{quant}'
            if adapter_path:
                model_name += f'-{get_model_name(adapter_path)}'
        if model_name in self.models.keys():
            if self.logger:
                self.logger.error(f'Model "{model_name}" already exists. If there are different configurations with the same model name, please use a different model name.')
                return
            else:
                raise ValueError(f'Model "{model_name}" already exists. If there are different configurations with the same model name, please use a different model name.')
        mlx_path = make_model_exist(model_id_or_path=model_path, quant=quant, revision=model_config.revision, adapter_path=adapter_path)
        mx.metal.clear_cache()
        self.models[model_name] = model_config

    def _switch_model(self, model_name: str) -> None:
        """Switch the model in ram to the given model.

        Args:
            model_name (str): Model name of the model.
        """
        if model_name not in self.models.keys():
            raise ValueError(f'No model named "{model_name}".')
        if model_name != self.current_model:
            del self.model, self.tokenizer, self.cache_manager, self.outline_model
            mx.metal.clear_cache()
            model_args = self.models[model_name]._asdict()
            model_args.pop('model_name')
            self.model, self.tokenizer = get_model_and_tokenizer(**model_args)
            self.current_model = model_name
            self.cache_manager = CacheManager(
                cache_dir=os.path.join(self.cache_dir, model_name),
                token_threshold=self.token_threshold,
                max_keep=self.max_keep,
                logger=self.logger
                )
            if OUTLINE_INSTALLED:
                self.outline_model = MlxLLM(model=self.model, tokenizer=self.tokenizer, cache_manager=self.cache_manager)
    
    def generate(self,
            model_name: str,
            prompt: str,
            stream: bool = False,
            stop: Optional[List[str]] = None,
            max_new_tokens: int = 256,
            temperature: float = 0,
            repetition_penalty: Optional[float] = None,
            repetition_context_size: Optional[int] = 100,
            top_p: float = 1.0,
            min_p: float = 0.0,
            min_tokens_to_keep: int = 1,
            logit_bias: Optional[Dict[int, float]] = None,
            prefill_step_size: Optional[int] = None,
            seed: Optional[int] = None,
            guided_json: Optional[Union[str, dict]] = None,
            guided_choice: Optional[List[str]] = None,
            guided_regex: Optional[str] = None,
            guided_grammar: Optional[str] = None,
            guided_whitespace_pattern: Optional[str] = None,
            **kwargs
        ) -> Union[EngineOutput, Iterator[EngineOutput]]:
        """Generate text with a model.

        Args:
            model_name (str): Name of the model to use.
            prompt (str): Text prompt for the generation.
            stream (bool, optional): Whether to return a generator for streaming text tokens during generation. Defaults to False.
            stop (Optional[List[str]], optional): List of texts to stop generation. Defaults to None.
            max_new_tokens (int, optional): Maximum number of tokens to generate. Defaults to 256.
            temperature (float, optional): Temperature that decide the "randomness" of the generation. Defaults to 0.
            repetition_penalty (Optional[float], optional): Penalty for repeated textt during generation. 1.0 being no penalty while higher values gives more penalty. Defaults to None.
            repetition_context_size (Optional[int], optional): Size of previous tokens to consider for repetition penalty. Defaults to 100.
            top_p (float, optional): Top P sampling. Defaults to 1.0.
            min_p (float, optional): Min P sampling. Defaults to 0.0.
            min_tokens_to_keep (int, optional): Minimum number of tokens to keep for sampling. Defaults to 1.
            logit_bias (Optional[Dict[int, float]], optional): Logits bias. Defaults to None.
            prefill_step_size (Optional[int], optional): Batch size for prompt preprocessing. Defaults to None.

        Returns:
            Union[str, Iterator[str]]: Generated text if `stream=False`.

        Yields:
            Iterator[Union[str, Iterator[str]]]: Generator of generated text if `stream=True`.
        """
        self._switch_model(model_name=model_name)
        guided_decoding = False
        if OUTLINE_INSTALLED and (guided_choice) is not None:
            guided_decoding = True
            gen_kwargs = dict(
                prompt=prompt, 
                logits_processor=get_choice_processor(guided_choice, self.outline_model),
                max_tokens=max_new_tokens,
                temp=temperature,
                top_p=top_p,
                seed=seed
            )
        elif OUTLINE_INSTALLED and (guided_json) is not None:
            guided_decoding = True
            gen_kwargs = dict(
                prompt=prompt, 
                logits_processor=get_json_processor(guided_json, self.outline_model, guided_whitespace_pattern),
                max_tokens=max_new_tokens,
                temp=temperature,
                top_p=top_p,
                seed=seed
            )
        elif OUTLINE_INSTALLED and (guided_regex) is not None:
            guided_decoding = True
            gen_kwargs = dict(
                prompt=prompt, 
                logits_processor=get_regex_processor(guided_regex, self.outline_model),
                max_tokens=max_new_tokens,
                temp=temperature,
                top_p=top_p,
                seed=seed
            )
        elif OUTLINE_INSTALLED and (guided_grammar) is not None:
            guided_decoding = True
            gen_kwargs = dict(
                prompt=prompt, 
                logits_processor=get_grammar_processor(guided_grammar, self.outline_model),
                max_tokens=max_new_tokens,
                temp=temperature,
                top_p=top_p,
                seed=seed
            )

        else:
            prompt_tokens = self.tokenizer.encode(prompt)
            prompt_tokens = remove_bos_duplicates(prompt_tokens, self.tokenizer.bos_token_id)
            cache, cache_id = self.cache_manager.find_cache(token_ids=prompt_tokens)
            gen_kwargs = dict(
                max_tokens=max_new_tokens,
                stop=stop,
                temp=temperature,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
                top_p=top_p,
                min_p=min_p,
                min_tokens_to_keep=min_tokens_to_keep,
                logit_bias=logit_bias,
                prefill_step_size=self.prefill_step_size if prefill_step_size is None else prefill_step_size,
                cache_history=cache,
                seed=seed,
                verbose=True
            )

        if stream:
            if not guided_decoding:
                def stream_output():
                    nonlocal cache
                    try:
                        for output in stream_generate(
                            model=self.model,
                            tokenizer=self.tokenizer,
                            prompt=prompt,
                            **gen_kwargs
                        ):
                            cache = output.step
                            yield EngineOutput(
                                text=output.text,
                                token=output.step.token,
                                token_ids=output.step.token_ids,
                                logprobs=output.step.logprobs,
                                prompt_len=output.prompt_len,
                                finish_reason=output.finish_reason
                            )
                    finally:
                        self.cache_manager.save_cache(cache=cache, from_cache_id=cache_id)
                        del cache, output
                        mx.metal.clear_cache()
            else:
                if self.logger:
                    self.logger.info('Using guided decoding...')
                def stream_output():
                    try:
                        for eo in self.outline_model.stream(**gen_kwargs):
                            yield eo
                    finally:
                        mx.metal.clear_cache()
            return stream_output()

        else:
            if not guided_decoding:
                output = generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    **gen_kwargs
                )
                cache = output.step
                engine_output = EngineOutput(
                        text=output.text,
                        token=output.step.token,
                        token_ids=output.step.token_ids,
                        logprobs=output.step.logprobs,
                        prompt_len=output.prompt_len,
                        finish_reason=output.finish_reason
                    )
                self.cache_manager.save_cache(cache=cache, from_cache_id=cache_id)
                del cache, output
                mx.metal.clear_cache()
            else:
                if self.logger:
                    self.logger.info('Using guided decoding...')
                engine_output = self.outline_model.generate(**gen_kwargs)
                mx.metal.clear_cache()
            return engine_output
