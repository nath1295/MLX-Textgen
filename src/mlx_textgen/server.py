import uvicorn
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from .engine import ModelEngine, ModelConfig
from .utils import PACKAGE_NAME
import asyncio
import argparse
import logging, json, uuid, warnings, yaml
from typing import Union, Dict, Any, Iterator, Literal, List, Tuple

warnings.filterwarnings('ignore')
logging.basicConfig(format='[(%(levelname)s) %(asctime)s]: %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Configure
def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f'{PACKAGE_NAME}.server',
        description='Run an OpenAI-compatible LLM server.',
    )
    parser.add_argument('-m', '--model-path', type=str, 
        default=None, help='Path to the model or the HuggingFace repository name if only one model should be served.')
    parser.add_argument('--tokenizer-path', type=str, 
        default=None, help='Path to the tokenizer or the HuggingFace repository name if only one model should be served. If None is given, it will be the model_path. Defaults to None.')
    parser.add_argument('--adapter-path', type=str, 
        default=None, help='Path to the adapter for the model. Defaults to None.')
    parser.add_argument('--revision', type=str, 
        default=None, help='Rivision of the repository if an HF repository is given. Defaults to None.')
    parser.add_argument('-q', '--quantize', type=str, 
        default='fp16', help='Model qunatization, options are "fp16", "q8", "q4", "q2". Defaults to "fp16", meaning no quantization.')
    parser.add_argument('--model-name', type=str,
        default=None, help='Model name appears in the API endpoint. If None is given, it will be created automatically with the model path. Defaults to None.')
    parser.add_argument('-cf', '--config-file', type=str, 
        default=None, 
        help='Path of the config file that store the configs of all models wanted to be served. If this is passed, "model-path", "quantize", and "model-name" will be ignored.')
    parser.add_argument('--prefill-step-size', type=int, 
        default=512, help='Batch size for model prompt processing. Defaults to 512.')
    parser.add_argument('-mk', '--max-keep', type=int, 
        default=10, help='Maximum number of cache history for each model to keep. Defaults to 10.')
    parser.add_argument('--token-threshold', type=int,
        default=50, 
        help='Minimum number of tokens in the prompt plus generated text to trigger prompt caching. Shorter prompts do not require caching to speed up generation. Defaults to 50.')
    parser.add_argument('--api-key', type=str, default=None, help='API key to access the endpoints. Defaults to None.')
    parser.add_argument('-p', '--port', type=int, 
                        default=5001, help='Port to server the API endpoints.')
    return parser

def parse_args() -> Tuple[ModelEngine, int, List[str]]:
    engine_args = get_arg_parser().parse_args()
    port = engine_args.port
    api_keys = [engine_args.api_key] if engine_args.api_key is not None else []
    if engine_args.config_file is not None:
        with open(engine_args.config_file, 'r') as f:
            model_args = yaml.safe_load(f)
        if isinstance(model_args, list):
            model_args = [ModelConfig(**args) for args in model_args]
        else:
            model_args = [ModelConfig(**model_args)]
    elif engine_args.model_path is not None:
        model_args = [
            ModelConfig(
                model_id_or_path=engine_args.model_path,
                tokenizer_id_or_path=engine_args.tokenizer_path,
                adapter_path=engine_args.adapter_path,
                quant=engine_args.quantize,
                revision=engine_args.revision,
                model_name=engine_args.model_name
            )
        ]
    else:
        raise ValueError('Either model_path or config_file has to be provide.')
    engine = ModelEngine(models=model_args, prefill_step_size=engine_args.prefill_step_size, 
        token_threshold=engine_args.token_threshold, max_keep=engine_args.max_keep, logger=logger)
    return engine, port, api_keys


engine, port, api_keys = parse_args()

models = [dict(id=m, object='model', created=1, owned_by='null', permission=[], root='root') for m in engine.models.keys()]
app = FastAPI()
semaphore = asyncio.Semaphore(1)

@app.get('/v1/models')
async def get_models() -> JSONResponse:
    return JSONResponse(content=jsonable_encoder(dict(object='list', data=models)))

def get_return_dict(text: Union[str, List[str]], id: str, model: str, return_type: Literal['text_completion', 'chat.completion'], stream: bool) -> Dict[str, Any]:
    if return_type == 'text_completion':
        choices = [dict(index=0, finish_reason='null', text=t) for t in text] if isinstance(text, list) else [dict(index=0, finish_reason='null', text=text)]
        return dict(id=id, 
            object=return_type, 
            created=1,
            model=model, 
            choices=choices
        )
    elif ((return_type == 'chat.completion') and (stream)):
        return dict(
            id="id", 
            object=return_type + '.chunk', 
            created=1, 
            model=model, 
            choices=[
                dict(
                    index=0, 
                    finish_reason="null", 
                    delta=dict(
                        role="assistant",
                        content=text)
                )
            ]
        )
    elif return_type == 'chat.completion':
        return dict(
            id=id, 
            object=return_type, 
            created=1, 
            model=model, 
            usage=dict(
                prompt_tokens=100, 
                completion_tokens=100, 
                total_tokens=200
            ), 
            choices=[
                dict(
                    index=0, 
                    message=dict(
                        role="assistant", 
                        content=text[0], 
                        tool_calls=[]
                    ), 
                    finish_reason="length"
                )
            ]
        )
    



# Generation
def stream_generate(args: Dict[str, Any]) -> Iterator[str]:
    extra_body = args.get('extra_body', dict())
    id = uuid.uuid4().hex
    response = engine.generate(
        model_name=args['model'],
        prompt=args['prompt'],
        stop=args.get('stop'),
        max_new_tokens=args.get('max_tokens', 1024),
        temperature=args.get('temperature', 0.0),
        top_p=args.get('top_p', 1),
        repetition_penalty=args.get('repetition_penalty', None),
        stream=True,
        **extra_body
    )
    for i in response:
        return_dict = get_return_dict(text=i, id=id, model=args['model'], return_type=args['endpoint'], stream=True)
        yield f'data: {json.dumps(return_dict)}\n\n'
    yield 'data: [DONE]'

    
def static_generate(args: Dict[str, Any]) -> Dict[str, Any]:
    extra_body = args.get('extra_body', dict())
    id = uuid.uuid4().hex
    if isinstance(args['prompt'], str):
        response = engine.generate(
            model_name=args['model'],
            prompt=args['prompt'],
            stop=args.get('stop'),
            max_new_tokens=args.get('max_tokens', 1024),
            temperature=args.get('temperature', 0.0),
            top_p=args.get('top_p', 1),
            repetition_penalty=args.get('frequency_penalty', None),
            stream=False,
            **extra_body
        )
        texts = [response]
    else:
        texts = []
        for prompt in args['prompt']:
            response = engine.generate(
                model_name=args['model'],
                prompt=prompt,
                stop=args.get('stop'),
                max_new_tokens=args.get('max_tokens', 1024),
                temperature=args.get('temperature', 0.0),
                top_p=args.get('top_p', 1),
                repetition_penalty=args.get('frequency_penalty', None),
                stream=False,
                **extra_body
            )
            texts.append(response)
    return_dict = get_return_dict(text=texts, id=id, model=args['model'], return_type=args['endpoint'], stream=False)
    return return_dict
    
async def async_generate_stream(args: Dict[str, Any]):
    generator = await asyncio.to_thread(stream_generate, args)
    for token in generator:
        yield token

async def async_generate(args: Dict[str, Any]):
    result = await asyncio.to_thread(static_generate, args)
    return result

async def async_chat_generate_stream(args: Dict[str, Any]):
    engine._switch_model(args['model'])
    if engine.tokenizer.chat_template is None:
            engine.tokenizer.chat_template = engine.tokenizer.default_chat_template
    prompt = engine.tokenizer.apply_chat_template(
            args['messages'], tokenize=True, add_generation_prompt=True
        )
    if ((len(prompt) > 0) and (prompt[0] == engine.tokenizer.bos_token_id)):
        prompt = prompt[1:]
    args['prompt'] = engine.tokenizer.decode(prompt, skip_special_tokens=False)
    generator = await asyncio.to_thread(stream_generate, args)
    for token in generator:
        yield token

async def async_chat_generate(args: Dict[str, Any]):
    engine._switch_model(args['model'])
    if engine.tokenizer.chat_template is None:
            engine.tokenizer.chat_template = engine.tokenizer.default_chat_template
    prompt = engine.tokenizer.apply_chat_template(
            args['messages'], tokenize=True, add_generation_prompt=True
        )
    if ((len(prompt) > 0) and (prompt[0] == engine.tokenizer.bos_token_id)):
        prompt = prompt[1:]
    args['prompt'] = engine.tokenizer.decode(prompt, skip_special_tokens=False)
    result = await asyncio.to_thread(static_generate, args)
    return result

@app.post('/v1/completions',  response_model=None)
async def completions(request: Request) -> Union[StreamingResponse, JSONResponse]:
    content = await request.json()
    content['endpoint'] = 'text_completion'
    api_key = request.headers.get('authorization', 'Bearer ').removeprefix('Bearer ')
    if ((api_key not in api_keys) and (len(api_keys) != 0)):
        return JSONResponse(jsonable_encoder(dict(error='Invalid API key.')), status_code=404)
    stream = content.get('stream', False)
    async with semaphore:
        if ((stream) and (isinstance(content['prompt'], str))):
            return StreamingResponse(async_generate_stream(content), media_type="text/event-stream")
        else:
            result = await async_generate(content)
            return JSONResponse(jsonable_encoder(result))
        
@app.post('/v1/chat/completions',  response_model=None)
async def completions(request: Request) -> Union[StreamingResponse, JSONResponse]:
    content = await request.json()
    content['endpoint'] = 'chat.completion'
    api_key = request.headers.get('authorization', 'Bearer ').removeprefix('Bearer ')
    if ((api_key not in api_keys) and (len(api_keys) != 0)):
        return JSONResponse(jsonable_encoder(dict(error='Invalid API key.')), status_code=404)
    stream = content.get('stream', False)
    async with semaphore:
        if stream:
            return StreamingResponse(async_chat_generate_stream(content), media_type="text/event-stream")
        else:
            result = await async_chat_generate(content)
            return JSONResponse(jsonable_encoder(result))
        
def main():
    uvicorn.run(app, port=port)

if __name__ == '__main__':
    main()
