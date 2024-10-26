import uvicorn
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse
from .engine import ModelEngine, TextCompletionInput, ChatCompletionInput
from .model_utils import PACKAGE_NAME, ModelConfig
import asyncio
import argparse
import logging, json, warnings, yaml
from typing import Union, Dict, Any, List, Tuple, Callable, Optional

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
        default=50, help='Maximum number of cache history for each model to keep. Defaults to 50.')
    parser.add_argument('--token-threshold', type=int,
        default=20, 
        help='Minimum number of tokens in the prompt plus generated text to trigger prompt caching. Shorter prompts do not require caching to speed up generation. Defaults to 20.')
    parser.add_argument('--api-key', type=str, default=None, help='API key to access the endpoints. Defaults to None.')
    parser.add_argument('-p', '--port', type=int, 
                        default=5001, help='Port to server the API endpoints.')
    parser.add_argument('--host', type=str, 
                        default='127.0.0.1', help='Host to bind the server to. Defaults to "127.0.0.1".')
    return parser

def parse_args() -> Tuple[ModelEngine, int, List[str]]:
    engine_args = get_arg_parser().parse_args()
    host = engine_args.host
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
    return engine, host, port, api_keys

def convert_arguments(new: str, old: str, args: Dict[str, Any], transform_fn: Optional[Callable] = None) -> Dict[str, Any]:
    value = args.pop(new, None)
    if value is not None:
        args[old] = value if transform_fn is None else transform_fn(value)
    return args

engine, host, port, api_keys = parse_args()

app = FastAPI()
semaphore = asyncio.Semaphore(1)
    
    
async def async_generate_stream(args: Dict[str, Any]):
    args['completion_type'] = 'text_completion'
    generator = await asyncio.to_thread(engine.generate, **args)
    try:
        for token in generator:
            yield f'data: {json.dumps(token)}\n\n'
        yield 'data: [DONE]'
    except Exception as e:
        logger.error(e)
        raise e

async def async_generate(args: Dict[str, Any]):
    args['completion_type'] = 'text_completion'
    result = await asyncio.to_thread(engine.generate, **args)
    return result

async def async_chat_generate_stream(args: Dict[str, Any]):
    generator = await asyncio.to_thread(engine.chat_generate, **args)
    try:
        for token in generator:
            yield f'data: {json.dumps(token)}\n\n'
        yield 'data: [DONE]'
    except asyncio.CancelledError:
        logger.error('Stopped in chat stream.')

async def async_chat_generate(args: Dict[str, Any]):
    result = await asyncio.to_thread(engine.chat_generate, **args)
    return result

@app.post('/v1/completions',  response_model=None)
async def completions(request: Request) -> Union[StreamingResponse, JSONResponse]:
    content = await request.json()
    api_key = request.headers.get('authorization', 'Bearer ').removeprefix('Bearer ')
    if ((api_key not in api_keys) and (len(api_keys) != 0)):
        return JSONResponse(jsonable_encoder(dict(error='Invalid API key.')), status_code=404)
    model = content.get('model')
    if model not in engine.models.keys():
        return JSONResponse(jsonable_encoder(dict(error=f'Model "{model}" does not exist.')), status_code=404)
    stream = content.get('stream', False)
    content = convert_arguments('max_completion_tokens', 'max_tokens', content)
    content = convert_arguments('frequency_penalty', 'repetition_penalty', content)
    content = convert_arguments('response_format', 'guided_json', content, transform_fn=lambda x: x.get('json_schema', dict()).get('schema'))
    if isinstance(content.get('stop', None), str):
        content['stop'] = [content['stop']]
    args_model = TextCompletionInput(**content)
    logger.info(args_model)
    args = args_model.model_dump()
    async with semaphore:
        if stream:
            return StreamingResponse(async_generate_stream(args), media_type="text/event-stream")
        else:
            result = await async_generate(args)
            return JSONResponse(jsonable_encoder(result))
        
@app.post('/v1/chat/completions',  response_model=None)
async def completions(request: Request) -> Union[StreamingResponse, JSONResponse]:
    content = await request.json()
    api_key = request.headers.get('authorization', 'Bearer ').removeprefix('Bearer ')
    if ((api_key not in api_keys) and (len(api_keys) != 0)):
        return JSONResponse(jsonable_encoder(dict(error='Invalid API key.')), status_code=404)
    model = content.get('model')
    if model not in engine.models.keys():
        return JSONResponse(jsonable_encoder(dict(error=f'Model "{model}" does not exist.')), status_code=404)
    stream = content.get('stream', False)
    content = convert_arguments('max_completion_tokens', 'max_tokens', content)
    content = convert_arguments('frequency_penalty', 'repetition_penalty', content)
    content = convert_arguments('response_format', 'guided_json', content, transform_fn=lambda x: x.get('json_schema', dict()).get('schema'))
    if isinstance(content.get('stop', None), str):
        content['stop'] = [content['stop']]
    args_model = ChatCompletionInput(**content)
    logger.info(args_model)
    args = args_model.model_dump()
    async with semaphore:
        if stream:
            return StreamingResponse(async_chat_generate_stream(args), media_type="text/event-stream")
        else:
            result = await async_chat_generate(args)
            return JSONResponse(jsonable_encoder(result))

@app.get('/v1/models')
async def get_models(request: Request) -> JSONResponse:
    api_key = request.headers.get('authorization', 'Bearer ').removeprefix('Bearer ')
    if ((api_key not in api_keys) and (len(api_keys) != 0)):
        return JSONResponse(jsonable_encoder(dict(error='Invalid API key.')), status_code=404)
    return JSONResponse(content=jsonable_encoder(dict(object='list', data=engine.model_info))) 

@app.get('/v1/models/{model_id}')
async def get_model(request: Request, model_id: str) -> JSONResponse:
    api_key = request.headers.get('authorization', 'Bearer ').removeprefix('Bearer ')
    if ((api_key not in api_keys) and (len(api_keys) != 0)):
        return JSONResponse(jsonable_encoder(dict(error='Invalid API key.')), status_code=404)
    model_dict = {info['id']: info for info in engine.model_info}
    if model_id not in model_dict.keys():
        return JSONResponse(jsonable_encoder(dict(error='Invalid model ID.')), status_code=404)
    return JSONResponse(content=jsonable_encoder(model_dict[model_id]))

def main():
    uvicorn.run(app, port=port, host=host)

if __name__ == '__main__':
    main()
