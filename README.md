# MLX-Textgen
[![PyPI](https://img.shields.io/pypi/v/mlx-textgen)](https://pypi.org/project/mlx-textgen/)
[![PyPI - License](https://img.shields.io/pypi/l/mlx-textgen)](https://pypi.org/project/mlx-textgen/)
[![GitHub Repo stars](https://img.shields.io/github/stars/nath1295/mlx-textgen)](https://pypi.org/project/mlx-textgen/)

## An OpenAI-compatible API LLM engine with smart prompt caching, batch processing, structured output with guided decoding, and function calling for all models using MLX  

MLX-Textgen is a light-weight LLM serving engine that utilize MLX and a smart KV cache management system to make your LLM generation more seamless on your Apple silicon machine. It features:
- **Multiple KV-cache slots to reduce the needs of prompt processing**
- **Structured text generation with json schemas, regex, or context free grammar**
- **Batch inferencing with multiple prompts**
- **Multiple models serving with Fastapi**
- **Common OpenAI API endpoints: `/v1/models`, `/v1/completions`, `/v1/chat/completions`**

It is built with:
1. [mlx-lm](https://github.com/ml-explore/mlx-examples)
2. [Outlines](https://github.com/dottxt-ai/outlines)
3. [FastAPI](https://github.com/fastapi/fastapi)

## Updates
- **2024-10-20:** Batch inference and function calling supported. Breaking changes in the save format for the KV caches. Run `mlx_textgen.clear_cache` after updating to avoid issues.
- **2024-10-07:** Guided decoding is supported with [Outlines](https://github.com/dottxt-ai/outlines) backend.

## Installing MLX-Textgen
MLX-textgen can be easily installed with `pip`:
```
pip install mlx-textgen
```

## Usage
### 1. Serving a single model
You can quickly set up a OpenAI API server with a single command.

```bash
mlx_textgen.server --model NousResearch/Hermes-3-Llama-3.1-8B --qunatize q8 --port 5001 --host 127.0.0.1
```

### 2. Serving a multiple models server
Create a config file template and add as many model as you like.
```bash
mlx_textgen.create_config --num-models 2
```

It will generate a file called `model_config.yaml`. Edit this file for the models you want to serve.
```yaml
- model_id_or_path: NousResearch/Hermes-3-Llama-3.1-8B
  tokenizer_id_or_path: null
  adapter_path: null
  quant: q8
  revision: null
  model_name: null
  model_config: null
  tokenizer_config: null
- model_id_or_path: mlx-community/Llama-3.2-3B-Instruct-4bit
  tokenizer_id_or_path: null
  adapter_path: null
  quant: q4
  revision: null
  model_name: llama-3.2-3b-instruct
  model_config: null
  tokenizer_config: null
```

Then start the engine:
```bash
mlx_textgen.server --config-file ./model_config.yaml --port 5001 --host 127.0.0.1
```

### 3. More engine arguments
You can check the details of other engine arguments by running:
```bash
mlx_textgen.server --help
```

You can specify the number of cache slots for each model, minimum number of tokens to create a cache file, and API keys etc.

## Features
### 1. Multiple KV cache slots support
All the KV cache are stored on disk. Therefore, unlike other LLM serving engine, a newly created KV cache will not overwrite the existing KV cache. This works better for agentic workflows where different types of prompts are being used frequently without losing previous cache for a long prompt.

### 2. Guided decoding with Regex, Json schema, and Grammar
You can pass your guided decoding argument `guided_json`, `guided_choice`, `guided_regex`, or `guided_grammar` as extra arguments and create structured text generation in a similar fashion to [vllm](https://github.com/vllm-project/vllm).

### 3. Batch inference support
Batch inference is supported for multiple prompts or multiple generations for a single prompt. Just pass a list of prompts to the `prompt` argument to the `/v1/completions` endpoint or `n=2` (or more than 2) to the `/v1/chat/completions` or `v1/completions` endpoints for batch inferencing.

### 4. Function calling support
Function calling with the `/v1/chat/completions` is supported. Simply use the `tools` and `tool_choice` arguments to supply lists of tools. There are three modes of using function calling:
1. `tool_choice="auto"`: The model will decide if tool calling is needed based on the conversation. If a tool is needed, it will pick the appropriate tool and generate the arguments. Otherwise, it will only response with normal text.
2. `tool_choice="required"`: One of the given tools must be selected by the model. The model will pick the appropriate tool and generate the arguments.
3. `tool_choice={"type": "function", "function": {"name": "<selected tool name>"}}`: The model will generate the arguments of the selected tools.  

If function calling is triggered, the call arguments will be contained in the `tool_calls` attribute in the `choices` element in the response. The `finish_reason` will be `tool_calls`.
```python
from openai import OpenAI

tools = [{
  "type": "function",
  "function": {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string"
        },
        "unit": {
          "type": "string",
          "default": "celsius"
        }
      },
      "required": ["location"]
    }
  }
}]

client = OpenAI(api_key='Your API Key', base_url='http://localhost:5001/v1/')

output = client.chat.completions.create(
    model='my_llama_model',
    messages=[
        dict(role='user', content='What is the current weather in London?')
    ],
    max_tokens=256,
    tools=tools,
    tool_choice='auto',
    stream=False
).choices[0].model_dump()

# output: 
# {'finish_reason': 'tool_calls',
#  'index': 0,
#  'logprobs': None,
#  'message': {'content': None,
#   'role': 'assistant',
#   'function_call': None,
#   'tool_calls': [{'id': 'call_052c8a6b',
#     'function': {'arguments': '{"location": "London", "unit": "celsius" }',
#      'name': 'get_current_weather'},
#     'type': 'function',
#     'index': 0}]}}
```

If `tool_choice="none"` is passed, the list of tools provided will be ignored and the model will only generate normal text.

### 5. Multiple LLMs serving
Only one model is loaded on ram at a time, but the engine leverage MLX fast module loading time to spin up another model when it is requested. This allows serving multiple models with one endpoint.

### 6. Automatic model quantisation
When configuring your model, you can specify the quantisation to increase your inference speed and lower memory usage. The original model is converted to MLX quantised model format when initialising the serving engine.

```python
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI(api_key='Your API Key', base_url='http://localhost:5001/v1/')

class Customer(BaseModel):
    first_name: str
    last_name: str
    age: int

prompt = """Extract the customer information from the following text in json format:
"...The customer David Stone join our membership in 20023, his current age is thirty five years old...."
"""
for i in client.chat.completions.create(
    model='my_llama_model',
    messages=[dict(role='user', content=prompt)],
    max_tokens=200,
    stream=True,
    extra_body=dict(
        guided_json=Customer.model_json_schema()
    )
):
    print(i.choices[0].delta.content, end='')

# Output: {"first_name": "David", "last_name": "Stone", "age": 35}
```

## License
This project is licensed under the terms of the MIT license.
