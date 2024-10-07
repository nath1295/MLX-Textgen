# MLX-Textgen
[![PyPI](https://img.shields.io/pypi/v/mlx-textgen)](https://pypi.org/project/mlx-textgen/)
[![PyPI - License](https://img.shields.io/pypi/l/mlx-textgen)](https://pypi.org/project/mlx-textgen/)
[![GitHub Repo stars](https://img.shields.io/github/stars/nath1295/mlx-textgen)](https://pypi.org/project/mlx-textgen/)

## A python package for serving LLM on OpenAI-compatible API endpoints with prompt caching using MLX  

MLX-Textgen is a light-weight LLM serving engine that utilize MLX and a smart KV cache management system to make your LLM generation more seamless on your Apple silicon machine. It features:
- Multiple KV-cache slots to reduce the needs of prompt processing
- Multiple models serving with Fastapi
- Common OpenAI API endpoints: `/v1/models`, `/v1/completions`, `/v1/chat/completions`

## Updates
**2024-10-07** - Guided decoding is supported with [Outlines](https://github.com/dottxt-ai/outlines) backend.

## Installing MLX-Textgen
MLX-textgen can be easily installed with `pip`:
```
pip install mlx-textgen
```

If you want guided decoding support for structured text generation, please install [Outlines](https://github.com/dottxt-ai/outlines) from source as well. Current realease of Outlines (`v0.0.46`) might not work.
```bash
git clone https://github.com/dottxt-ai/outlines.git;
cd outlines;
pip install -U .;
```
You might need to install Pytorch as a dependency of Outlines as well.

## Features
### 1. Multiple KV cache slots support
All the KV cache are stored on disk. Therefore, unlike other LLM serving engine, a newly created KV cache will not overwrite the existing KV cache. This works better for agenic workflows where different types of prompts are being used frequently without losing previous cache for a long prompt.

### 2. Multiple LLMs serving
Only one model is loaded on ram at a time, but the engine leverage MLX fast module loading time to spin up another model when it is requested. This allows serving multiple models with one endpoint.

### 3. Automatic model quantisation
When configuring your model, you can specify the quantisation to increase your inference speed and lower memory usage. The original model is converted to MLX quantised model format when initialising the serving engine.

### 4. Guided decoding with Regex, Json schema, and Grammar
If [Outlines](https://github.com/dottxt-ai/outlines) is installed with the recommended way, guided decoding is supported. If you are using the `openai` package in python, you can pass your guided decoding argument `guided_json`, `guided_choice`, `guided_regex`, or `guided_grammar` as extra arguments and create structured generation in a similar fashion to [vllm](https://github.com/vllm-project/vllm).

```python
from pydantic import BaseModel
from openai import Client

client = Client(api_key='Your API Key', base_url='http://localhost:5001/v1/')

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

## Usage
### 1. Serving a single model
You can quickly set up a OpenAI API server with a single command.

```bash
mlx_textgen.server --model NousResearch/Hermes-3-Llama-3.1-8B --qunatize q8 --port 5001
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
mlx_textgen.server --config-file ./model_config.yaml --port 5001
```

### 3. More engine arguments
You can check the details of other engine arguments by running:
```bash
mlx_textgen.server --help
```

You can specify the number of cache slots for each model, minimum number of tokens to create a cache file, and API keys etc.

## License
This project is licensed under the terms of the MIT license.
