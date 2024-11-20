import os
from mlx_lm.utils import get_model_path, load_config, load_model, convert
from .tokenizer_utils import load_tokenizer, TokenizerWrapper
from logging import Logger
import mlx.nn as nn
from typing import Literal, Optional, Dict, Any, Tuple, NamedTuple
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

PACKAGE_NAME = 'mlx_textgen'

package_cache_dir = os.path.join(os.path.expanduser('~'), '.cache', PACKAGE_NAME)
os.makedirs(package_cache_dir, exist_ok=True)

def clear_prompt_cache() -> None:
    prompt_cache_dir = os.path.join(package_cache_dir, 'prompt_cache')
    if os.path.exists(prompt_cache_dir):
        from shutil import rmtree
        rmtree(prompt_cache_dir)

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
    mlx_path = os.path.join(package_cache_dir, model_base_name)
    
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
                mlx_path = os.path.join(package_cache_dir, model_base_name + f'-q{quant_target}')
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
        tokenizer_config: Optional[Dict[str, Any]] = None,
        logger: Optional[Logger] = None
    ) -> Tuple[nn.Module, TokenizerWrapper]:
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
        Tuple[nn.Module, TokenizerWrapper]: The model and the tokenizer.
    """
    mlx_path = make_model_exist(model_id_or_path=model_id_or_path, quant=quant, revision=revision, adapter_path=adapter_path)
    tokenizer_config = {} if tokenizer_config is None else tokenizer_config
    model_config = {} if model_config is None else model_config
    model = load_model(model_path=mlx_path, model_config=model_config)
    tokenizer = load_tokenizer(model_path=mlx_path if tokenizer_id_or_path is None else Path(tokenizer_id_or_path), tokenizer_config_extra=tokenizer_config, logger=logger)
    return model, tokenizer

class ModelConfig(NamedTuple):
    model_id_or_path: str
    tokenizer_id_or_path: Optional[str] = None
    adapter_path: Optional[str] = None
    quant: Literal['fp16', 'q8', 'q4', 'q2'] = 'fp16'
    revision: Optional[str] = None
    model_name: Optional[str] = None
    model_config: Optional[Dict[str, Any]] = None
    tokenizer_config: Optional[Dict[str, Any]] = None