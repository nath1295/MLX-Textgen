from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx.nn import Module
from datetime import datetime as dt
import time
import uuid
import mlx.core as mx
from .cache_utils import CacheHistory, make_empty_cache
from .model_utils import package_cache_dir, get_model_and_tokenizer, ModelConfig, get_model_name, make_model_exist
from .generate_utils import (
    generate, 
    stream_generate, 
    remove_bos_duplicates,
    get_choice_processor, 
    get_json_processor, 
    get_regex_processor, 
    get_grammar_processor,
    GenerationOutput, 
    OUTLINE_INSTALLED
    )
from .chat_utils import ChatTemplate, convert_tool_to_json_schema, get_tool_name
from .caching import CacheManager
from logging import Logger
from pydantic import BaseModel, Field, field_validator
import os
from typing import Optional, List, Dict, Union, Literal, Any, Iterator, Generator
if OUTLINE_INSTALLED:
    from outlines.models.transformers import TransformerTokenizer
    from outlines.processors.base_logits_processor import OutlinesLogitsProcessor

def get_return_dict(
        generation_outputs: List[GenerationOutput], 
        id: str, model: str, 
        completion_type: Literal['text_completion', 'chat.completion'], 
        created: float,
        stream: bool,
        indices: Optional[List[int]] = None,
        tool_start: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_arguments: Optional[str] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    num_outputs = len(generation_outputs)
    indices = indices if indices else range(num_outputs)
    if num_outputs != len(indices):
        raise Exception('Indices length mistmatch.')
    if completion_type == 'text_completion':
        choices = [dict(index=i, finish_reason=go.finish_reason if go.stop_text != tool_start else 'tool_call_start', text=go.text) for i, go in zip(indices, generation_outputs)]
        if stream:
            out_dicts = [dict(
            id='cmpl-'+id, 
            object=completion_type, 
            created=created,
            model=model,
            choices=[c]
            ) for c in choices]
        else:
            tokens = [len(go.token_ids.tolist()) for go in generation_outputs]
            out_dicts = dict(id='cmpl-'+id, 
            object=completion_type, 
            created=created,
            model=model, 
            choices=choices,
            usage=dict(
                total_tokens=sum(tokens)
            )
        )
        return out_dicts
    elif ((completion_type == 'chat.completion') and (stream)):
        return [dict(
            id='chatcmpl-'+id, 
            object=completion_type + '.chunk', 
            created=created, 
            model=model, 
            choices=[
                dict(
                    index=i, 
                    finish_reason=(go.finish_reason if go.stop_text != tool_start else 'tool_call_start') if tool_call_id is None else 'tool_calls', 
                    delta=dict(
                        role="assistant",
                        content=go.text if tool_call_id is None else None,
                        tool_calls=[] if tool_call_id is None else [
                                {
                                    "index": 0,
                                    "id": f'call_{tool_call_id}',
                                    "function": {
                                        "arguments": tool_arguments,
                                        "name": tool_name
                                    },
                                    "type": 'function'
                                }
                            ]
                    )
                )
            ]
        ) for i, go in zip(indices, generation_outputs)]
    elif completion_type == 'chat.completion':
        tokens = [len(go.token_ids.tolist()) for go in generation_outputs]
        return dict(
            id='chatcmpl-'+id, 
            object=completion_type, 
            created=created, 
            model=model, 
            usage=dict(
                total_tokens=sum(tokens)
            ), 
            choices=[
                dict(
                    index=i, 
                    message=dict(
                        role="assistant", 
                        content=go.text if tool_call_id is None else None,
                        tool_calls=[] if tool_call_id is None else [
                                {
                                    "index": 0,
                                    "id": f'call_{tool_call_id}',
                                    "function": {
                                        "arguments": tool_arguments,
                                        "name": tool_name
                                    },
                                    "type": 'function'
                                }
                            ]
                    ),
                    finish_reason=(go.finish_reason if go.stop_text != tool_start else 'tool_call_start') if tool_call_id is None else 'tool_calls'
                )
            for i, go in enumerate(generation_outputs)]
        )

class TextCompletionInput(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    stream: bool = False
    stop: Optional[List[str]] = None
    max_tokens: int = Field(default=4096, ge=1, le=32000)
    n: int = Field(default=1, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    repetition_penalty: Optional[float] = Field(default=None, gt=0.0, le=2.0)
    repetition_context_size: Optional[int] = Field(default=100, ge=1)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    min_tokens_to_keep: int = Field(default=1, ge=1)
    logit_bias: Optional[Dict[int, float]] = None
    seed: Optional[int] = None
    guided_json: Optional[Union[str, dict]] = None
    guided_choice: Optional[List[str]] = None
    guided_regex: Optional[str] = None
    guided_grammar: Optional[str] = None
    guided_whitespace_pattern: Optional[str] = None

    @field_validator('prompt')
    def check_prompt(cls, v):
        if isinstance(v, str) and not v.strip():
            raise ValueError('Prompt cannot be an empty string.')
        if isinstance(v, list) and not v:
            raise ValueError('Prompt cannot be an empty list.')
        return v

class ChatCompletionInput(BaseModel):
    model: str
    messages: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]
    stream: bool = False
    stop: Optional[List[str]] = None
    max_tokens: int = Field(default=4096, ge=1, le=32000)
    n: int = Field(default=1, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    repetition_penalty: Optional[float] = Field(default=None, gt=0.0, le=2.0)
    repetition_context_size: Optional[int] = Field(default=100, ge=1)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    min_tokens_to_keep: int = Field(default=1, ge=1)
    logit_bias: Optional[Dict[int, float]] = None
    seed: Optional[int] = None
    guided_json: Optional[Union[str, dict]] = None
    guided_choice: Optional[List[str]] = None
    guided_regex: Optional[str] = None
    guided_grammar: Optional[str] = None
    guided_whitespace_pattern: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Union[Literal['none', 'auto', 'required'], Dict[str, Union[str, Dict[str, str]]]] = 'auto'

    @field_validator('messages')
    def check_messages(cls, v):
        if isinstance(v, list) and len(v) == 0:
            raise ValueError('Messages cannot be an empty list.')
        for msgs in v:
            if len(msgs) == 0:
                raise ValueError('Messages cannot be an empty list.')
        return v

class ModelEngine:

    def __init__(self, 
            models: Union[ModelConfig, List[ModelConfig]],
            prefill_step_size: int = 512,
            token_threshold: int = 20,
            save_similarity_threshold: float = 0.9,
            max_keep: int = 50,
            verbose: bool = True,
            logger: Optional[Logger] = None
        ) -> None:
        """Initialising the engine.

        Args:
            models (Union[ModelConfig, List[ModelConfig]]): Model configurations or list of model configurations to serve.
            prefill_step_size (int, optional): Batch size for prompt preprocessing. Defaults to 512.
            token_threshold (int, optional): Minimum number of tokens to be considered to save as a prompt cache history. Defaults to 20.
            max_keep (int, optional): Maximum number of cache history files to keep. Defaults to 50.
            logger (Optional[Logger], optional): If a logger is provided, messages of adding and deleting caches will be added.
        """
        self.model_configs = [models] if isinstance(models, ModelConfig) else models
        self.prefill_step_size = prefill_step_size
        self.token_threshold = token_threshold
        self.save_sim_threshold = save_similarity_threshold
        self.max_keep = max_keep
        self.verbose = verbose
        self.logger = logger
        self.models: Dict[str, ModelConfig] = dict()
        self.model: Optional[Module] = None
        self.tokenizer: Optional[TokenizerWrapper] = None
        self.current_model: Optional[str] = None
        self.cache_manager: Optional[CacheManager] = None
        self.outlines_tokenizer: Optional[TransformerTokenizer] = None
        self.chat_template: Optional[ChatTemplate] = None
        self.cache_dir = os.path.join(package_cache_dir, 'prompt_cache')
        for model_config in self.model_configs:
            self._prepare_model(model_config=model_config)
        self.model_info = [dict(id=m, object='model', created=int(dt.now().timestamp()), owned_by=None, permission=[], root='root') for m in self.models.keys()]

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

    def _log(self, msg: str) -> None:
        if self.logger:
            self.logger.info(msg)

    def _switch_model(self, model_name: str) -> None:
        """Switch the model in ram to the given model.

        Args:
            model_name (str): Model name of the model.
        """
        if model_name not in self.models.keys():
            raise ValueError(f'No model named "{model_name}".')
        if model_name != self.current_model:
            start = time.perf_counter()
            del self.model, self.tokenizer, self.cache_manager, self.outlines_tokenizer
            mx.metal.clear_cache()
            model_args = self.models[model_name]._asdict()
            model_args.pop('model_name')
            self.model, self.tokenizer = get_model_and_tokenizer(**model_args)
            self.current_model = model_name
            model_key = get_model_name(self.models[model_name].model_id_or_path)
            self.cache_manager = CacheManager(
                cache_dir=os.path.join(self.cache_dir, model_key),
                token_threshold=self.token_threshold,
                save_similarity_threshold=self.save_sim_threshold,
                max_keep=self.max_keep,
                logger=self.logger
                )
            self.chat_template = ChatTemplate(tokenizer=self.tokenizer._tokenizer)
            if OUTLINE_INSTALLED:
                self.outlines_tokenizer = TransformerTokenizer(tokenizer=self.tokenizer._tokenizer)
            end = time.perf_counter()
            self._log(f'Switch to model "{model_name}"; time taken: {end - start:.4f}s')

    def _get_outlines_processor(self,
            guided_json: Optional[Union[str, dict]] = None,
            guided_choice: Optional[List[str]] = None,
            guided_regex: Optional[str] = None,
            guided_grammar: Optional[str] = None,
            guided_whitespace_pattern: Optional[str] = None
            ) -> Optional[OutlinesLogitsProcessor]:
        logits_processor = None
        if OUTLINE_INSTALLED and (guided_choice) is not None:
            logits_processor=get_choice_processor(guided_choice, self.outlines_tokenizer)

        elif OUTLINE_INSTALLED and (guided_json) is not None:
            logits_processor=get_json_processor(guided_json, self.outlines_tokenizer, guided_whitespace_pattern)

        elif OUTLINE_INSTALLED and (guided_regex) is not None:
            logits_processor=get_regex_processor(guided_regex, self.outlines_tokenizer)
            
        elif OUTLINE_INSTALLED and (guided_grammar) is not None:
            logits_processor = get_grammar_processor(guided_grammar, self.outlines_tokenizer)

        return logits_processor

    def generate(self,
            model: str,
            prompt: Union[str, List[str]],
            completion_type: Literal['text_completion', 'chat.completion'] = 'text_completion', 
            stream: bool = False,
            stop: Optional[List[str]] = None,
            max_tokens: int = 4096,
            n: int = 1,
            temperature: float = 0,
            repetition_penalty: Optional[float] = None,
            repetition_context_size: Optional[int] = 100,
            top_p: float = 1.0,
            min_p: float = 0.0,
            min_tokens_to_keep: int = 1,
            logit_bias: Optional[Dict[int, float]] = None,
            seed: Optional[int] = None,
            guided_json: Optional[Union[str, dict]] = None,
            guided_choice: Optional[List[str]] = None,
            guided_regex: Optional[str] = None,
            guided_grammar: Optional[str] = None,
            guided_whitespace_pattern: Optional[str] = None,
            # chat specific arguments
            tools: Optional[List[Dict[str, Any]]] = None,
            tool_choice: Union[Literal['none','auto', 'required'], Dict[str, Union[str, Dict[str, str]]]] = 'auto',
            **kwargs
        ) -> Union[List[GenerationOutput], Iterator[GenerationOutput]]:
        """Generate text with a model.

        Args:
            model_name (str): Name of the model to use.
            prompt (str): Text prompt for the generation.
            stream (bool, optional): Whether to return a generator for streaming text tokens during generation. Defaults to False.
            stop (Optional[List[str]], optional): List of texts to stop generation. Defaults to None.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 256.
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
        self._switch_model(model_name=model)
        prompts = [prompt] if isinstance(prompt, str) else prompt
        prompt_tokens = [remove_bos_duplicates(self.tokenizer.encode(p), bos_token_id=self.tokenizer.bos_token_id) for p in prompts]
        if n != 1:
            prompt_tokens = sum([[p] * n  for p in prompt_tokens], [])
            prompts = sum([[p] * n  for p in prompts], [])
        num_prompts = len(prompts)
        prompt_index = list(range(num_prompts))
        cache_history, cache_ids = self.cache_manager.find_cache(token_ids=prompt_tokens)
        cache_history = CacheHistory(cache=make_empty_cache(model=self.model), token_ids=[]) if cache_history is None else cache_history

        stop = [stop] if isinstance(stop, str) else stop
        
        logits_processor = self._get_outlines_processor(guided_json=guided_json, 
            guided_choice=guided_choice, guided_regex=guided_regex, guided_grammar=guided_grammar, guided_whitespace_pattern=guided_whitespace_pattern)
        
        # Handling tool calls
        tool_stage = None
        tool_name = None
        if tools:
            if tool_choice == 'auto':
                if isinstance(stop, list):
                    stop.append(self.chat_template.tool_start)
                else:
                    stop = [self.chat_template.tool_start]
            elif tool_choice == 'required':
                tool_stage = 'pick_tool'
                tool_names = [get_tool_name(tool) for tool in tools if get_tool_name(tool) is not None]
                logits_processor = self._get_outlines_processor(guided_choice=tool_names)
            elif tool_choice == 'none':
                pass
            else:
                tool_stage = 'gen_args'
                tool_name = tool_choice['function']['name']
                tool_dict = list(filter(lambda x: get_tool_name(x) == tool_name, tools))[0]
                logits_processor = self._get_outlines_processor(guided_json=convert_tool_to_json_schema(tool_dict))

        mx.metal.clear_cache()
        
        gen_kwargs = dict(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompts,
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            min_tokens_to_keep=min_tokens_to_keep,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            cache_history=cache_history,
            prefill_batch_size=self.prefill_step_size,
            logit_bias=logit_bias,
            seed=seed,
            logits_processor=logits_processor,
            verbose=self.verbose,
            logger=self.logger
        )
        cpl_id = uuid.uuid4().hex
        created = int(dt.now().timestamp())
        if stream:
            # Streaming without tool call
            def stream_output() -> Generator[GenerationOutput, None, None]:
                nonlocal cache_history
                try:
                    stopped = []
                    is_completed = False
                    for go in stream_generate(**gen_kwargs):
                        return_dict = get_return_dict(generation_outputs=go, 
                                id=cpl_id, 
                                model=model, 
                                completion_type=completion_type, 
                                created=created, 
                                stream=stream,
                                indices=prompt_index,
                                tool_start=self.chat_template.tool_start
                            )
                        for g in return_dict:
                            index = g['choices'][0]['index']
                            finish_reason = g['choices'][0]['finish_reason']
                            if finish_reason in ['stop', 'tool_call_start', 'tool_calls']:
                                stopped.append(index)
                            if (index not in stopped) or (finish_reason in ['stop', 'tool_call_start', 'tool_calls']):
                                if finish_reason not in ['stop', 'length']:
                                    g['choices'][0]['finish_reason'] = None
                                yield g
                    is_completed = True
                finally:
                    # To avoid weird behavior if the competion is not completed
                    if is_completed:
                        token_ids = [g.token_ids.tolist() for g in go]
                        new_ch = CacheHistory(cache=cache_history.cache, token_ids=token_ids)
                        self.cache_manager.save_cache(cache=new_ch)
                        del cache_history, new_ch
                    mx.metal.clear_cache()
            
            # Streaming with tool call
            def tool_stream_output():
                if num_prompts != 1:
                    raise Exception(f'Tool calling for more than one chain of messages is not supported.')
                if completion_type != 'chat.completion':
                    raise Exception('Tool calling only supported for chat completion.')
                
                nonlocal cache_history, tool_stage, tool_name
                try:
                    is_completed = False
                    text = ''
                    call_id = uuid.uuid4().hex[:8]

                    # Allow generation if tool_choice is 'auto'
                    if tool_choice == 'auto':
                        for go in stream_generate(**gen_kwargs):
                            g = get_return_dict(generation_outputs=go, 
                                    id=cpl_id, 
                                    model=model, 
                                    completion_type=completion_type, 
                                    created=created, 
                                    stream=stream,
                                    indices=prompt_index,
                                    tool_start=self.chat_template.tool_start
                                )[0]
                            index = g['choices'][0]['index']
                            finish_reason = g['choices'][0]['finish_reason']
                            if finish_reason is None or (finish_reason in ('stop', 'length')):
                                text += g['choices'][0]['delta']['content']
                                yield g
                            elif finish_reason == 'tool_call_start':
                                tool_stage = 'pick_tool'
                                g['choices'][0]['finish_reason'] = None
                                # update the generation kwargs here
                                gen_kwargs['prompt'][0] += text + self.chat_template.tool_start + '{"function": {' + '"name": "'
                                token_ids=[go[0].token_ids.tolist()]
                                self.cache_manager.save_cache(cache=CacheHistory(cache=cache_history.cache, token_ids=token_ids))
                                mx.metal.clear_cache()
                                cache_history, cache_ids = self.cache_manager.find_cache([
                                    remove_bos_duplicates(self.tokenizer.encode(gen_kwargs['prompt'][0]), bos_token_id=self.tokenizer.bos_token_id)
                                ])
                                cache_history = CacheHistory(cache=make_empty_cache(model=self.model), token_ids=[]) if cache_history is None else cache_history
                                gen_kwargs['cache_history'] = cache_history
                                tool_names = [get_tool_name(tool) for tool in tools if get_tool_name(tool) is not None]
                                gen_kwargs['logits_processor'] = self._get_outlines_processor(guided_choice=tool_names)
                                yield g

                    # stage 2, generate tool name
                    if tool_stage == 'pick_tool':
                        go = generate(**gen_kwargs)
                        finish_reason = go[0].finish_reason
                        tool_name = go[0].text
                        if finish_reason == 'stop':
                            tool_stage = 'gen_args'
                            gen_kwargs['prompt'][0] += tool_name + '", arguments": '
                            token_ids=[go[0].token_ids.tolist()]
                            self.cache_manager.save_cache(cache=CacheHistory(cache=cache_history.cache, token_ids=token_ids))
                            mx.metal.clear_cache()
                            # Find cache again
                            cache_history, cache_ids = self.cache_manager.find_cache([
                                remove_bos_duplicates(self.tokenizer.encode(gen_kwargs['prompt'][0]), bos_token_id=self.tokenizer.bos_token_id)
                            ])
                            cache_history = CacheHistory(cache=make_empty_cache(model=self.model), token_ids=[]) if cache_history is None else cache_history
                            gen_kwargs['cache_history'] = cache_history
                            tool_dict = list(filter(lambda x: get_tool_name(x) == tool_name, tools))[0]
                            gen_kwargs['logits_processor'] = self._get_outlines_processor(guided_json=convert_tool_to_json_schema(tool_dict))

                    # stage 3, generate arugments
                    if tool_stage == 'gen_args':
                        go = generate(**gen_kwargs)
                        finish_reason = go[0].finish_reason
                        tool_arguments = go[0].text
                        return_dict = get_return_dict(
                            generation_outputs=go,
                            id=cpl_id, 
                            model=model, 
                            completion_type=completion_type, 
                            created=created, 
                            stream=stream,
                            indices=prompt_index,
                            tool_start=self.chat_template.tool_start,
                            tool_call_id=call_id,
                            tool_name=tool_name,
                            tool_arguments=tool_arguments
                        )
                        yield return_dict[0]
                    is_completed = True

                finally:
                    # To avoid weird behavior if the competion is not completed
                    if is_completed:
                        token_ids = [g.token_ids.tolist() for g in go]
                        new_ch = CacheHistory(cache=cache_history.cache, token_ids=token_ids)
                        self.cache_manager.save_cache(cache=new_ch)
                        del cache_history, new_ch
                    mx.metal.clear_cache()

            return stream_output() if not tools or tool_choice == 'none' else tool_stream_output()

        else:
            call_id = uuid.uuid4().hex[:8]
            tool_name = None
            tool_arguments = None
            if not tools or tool_choice in ['none', 'auto']:
                go = generate(**gen_kwargs)
                token_ids = [g.token_ids.tolist() for g in go]
                text = go[0].text
                finish_reason = go[0].finish_reason
                if (finish_reason == 'stop') and (go[0].stop_text == self.chat_template.tool_start):
                    tool_stage = 'pick_tool'
                    # update the generation kwargs here
                    gen_kwargs['prompt'][0] += text + self.chat_template.tool_start + '{"function": {' + '"name": "'
                    token_ids=[go[0].token_ids.tolist()]
                    self.cache_manager.save_cache(cache=CacheHistory(cache=cache_history.cache, token_ids=token_ids))
                    mx.metal.clear_cache()
                    cache_history, cache_ids = self.cache_manager.find_cache([
                        remove_bos_duplicates(self.tokenizer.encode(gen_kwargs['prompt'][0]), bos_token_id=self.tokenizer.bos_token_id)
                    ])
                    cache_history = CacheHistory(cache=make_empty_cache(model=self.model), token_ids=[]) if cache_history is None else cache_history
                    gen_kwargs['cache_history'] = cache_history
                    tool_names = [get_tool_name(tool) for tool in tools if get_tool_name(tool) is not None]
                    gen_kwargs['logits_processor'] = self._get_outlines_processor(guided_choice=tool_names)
                else:
                    call_id = None

            # stage 2, generate tool name
            if tool_stage == 'pick_tool':
                go = generate(**gen_kwargs)
                finish_reason = go[0].finish_reason
                tool_name = go[0].text
                if finish_reason == 'stop':
                    tool_stage = 'gen_args'
                    gen_kwargs['prompt'][0] += tool_name + '", arguments": '
                    token_ids=[go[0].token_ids.tolist()]
                    self.cache_manager.save_cache(cache=CacheHistory(cache=cache_history.cache, token_ids=token_ids))
                    mx.metal.clear_cache()
                    # Find cache again
                    cache_history, cache_ids = self.cache_manager.find_cache([
                        remove_bos_duplicates(self.tokenizer.encode(gen_kwargs['prompt'][0]), bos_token_id=self.tokenizer.bos_token_id)
                    ])
                    cache_history = CacheHistory(cache=make_empty_cache(model=self.model), token_ids=[]) if cache_history is None else cache_history
                    gen_kwargs['cache_history'] = cache_history
                    tool_dict = list(filter(lambda x: get_tool_name(x) == tool_name, tools))[0]
                    gen_kwargs['logits_processor'] = self._get_outlines_processor(guided_json=convert_tool_to_json_schema(tool_dict))

            # stage 3, generate arugments
            if tool_stage == 'gen_args':
                go = generate(**gen_kwargs)
                finish_reason = go[0].finish_reason
                tool_arguments = go[0].text

            token_ids=[go[0].token_ids.tolist()]
            self.cache_manager.save_cache(cache=CacheHistory(cache=cache_history.cache, token_ids=token_ids))
            del cache_history
            mx.metal.clear_cache()

            return get_return_dict(go, id=cpl_id, model=model, 
                completion_type=completion_type, created=created, stream=False, 
                tool_start=self.chat_template.tool_start, tool_call_id=call_id, tool_name=tool_name, tool_arguments=tool_arguments)
        
    def chat_generate(self,
            model: str,
            messages: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
            stream: bool = False,
            stop: Optional[List[str]] = None,
            max_tokens: int = 4096,
            n: int = 1,
            temperature: float = 0,
            repetition_penalty: Optional[float] = None,
            repetition_context_size: Optional[int] = 100,
            top_p: float = 1.0,
            min_p: float = 0.0,
            min_tokens_to_keep: int = 1,
            logit_bias: Optional[Dict[int, float]] = None,
            seed: Optional[int] = None,
            guided_json: Optional[Union[str, dict]] = None,
            guided_choice: Optional[List[str]] = None,
            guided_regex: Optional[str] = None,
            guided_grammar: Optional[str] = None,
            guided_whitespace_pattern: Optional[str] = None,
            tools: Optional[List[Dict[str, Any]]] = None,
            tool_choice: Union[Literal['none', 'auto', 'required'], Dict[str, Union[str, Dict[str, str]]]] = 'auto',
            **kwargs
        ) -> Union[List[GenerationOutput], Iterator[GenerationOutput]]:
        """Generate text with a model.

        Args:
            model_name (str): Name of the model to use.
            prompt (str): Text prompt for the generation.
            stream (bool, optional): Whether to return a generator for streaming text tokens during generation. Defaults to False.
            stop (Optional[List[str]], optional): List of texts to stop generation. Defaults to None.
            max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 256.
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
        self._switch_model(model_name=model)
        if not isinstance(messages[0], list):
            messages = [messages]
        prompt = [self.chat_template.apply_chat_template(messages=msgs, tools=tools, tool_choice=tool_choice) for msgs in messages]
        output = self.generate(
            model=model,
            prompt=prompt,
            completion_type='chat.completion',
            stream=stream,
            stop=stop,
            max_tokens=max_tokens,
            n=n,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            top_p=top_p,
            min_p=min_p,
            min_tokens_to_keep=min_tokens_to_keep,
            logit_bias=logit_bias,
            seed=seed,
            guided_json=guided_json,
            guided_choice=guided_choice,
            guided_regex=guided_regex,
            guided_grammar=guided_grammar,
            guided_whitespace_pattern=guided_whitespace_pattern,
            tools=tools,
            tool_choice=tool_choice
        )
        return output

    
