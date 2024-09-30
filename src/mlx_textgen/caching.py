from .utils import CacheHistory, StepOutput, save_cache, load_cache, find_max_prefix_num
import json, os
from logging import Logger
from datetime import datetime as dt
from typing import Optional, List, Union, Dict, NamedTuple, Tuple

class PromptSummary(NamedTuple):
    token_ids: List[int]
    length: int
    last_modified: dt

def prompt_summary_to_dict(ps: PromptSummary) -> Dict[str, Union[List[int], float]]:
    return dict(token_ids=ps.token_ids, last_modified=ps.last_modified.timestamp())

class CacheManager:
    """Class to manage multiple cache histories.
    """
    def __init__(self, cache_dir: str, token_threshold: int = 50, max_keep: int = 10, logger: Optional[Logger] = None) -> None:
        """Initialise the cache manager.

        Args:
            cache_dir (str): The directory to save all the caches.
            token_threshold (int, optional): Minimum number of tokens to be considered to save as a prompt cache history. Defaults to 50.
            max_keep (int, optional): Maximum number of cache history files to keep. Defaults to 10.
            logger (Optional[Logger], optional): If a logger is provided, messages of adding and deleting caches will be added.
        """
        self.cache_dir = os.path.abspath(cache_dir)
        self.json_dir = os.path.join(self.cache_dir, 'cached_prompts.json')
        self.token_threshold = token_threshold
        self.max_keep = max_keep
        self.logger = logger
        if not os.path.exists(self.json_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            prompts = dict()
            with open(self.json_dir, 'w') as f:
                json.dump(prompts, f)

    @property
    def cached_prompts(self) -> Dict[str, PromptSummary]:
        """Dictionary that contains the information of all the currently stored cache histories.

        Returns:
            Dict[str, PromptSummary]: Dictionary that contains the information of all the currently stored cache histories.
        """
        with open(self.json_dir, 'r') as f:
            prompts = json.load(f)
        for k, v in prompts.items():
            v['last_modified'] = dt.fromtimestamp(v['last_modified'])
            v['length'] = len(v['token_ids'])
            prompts[k] = PromptSummary(**v)   
        return prompts
    
    @property
    def new_cache_id(self) -> str:
        """An unused cache ID for a new cache history.

        Returns:
            str: An unused cache ID for a new cache history.
        """
        current_ids: List[str] = list(self.cached_prompts.keys())
        current_ids = [int(x.removeprefix('cache_')) for x in current_ids]
        if len(current_ids) == 0:
            return 'cache_0'
        max_id = max(current_ids)
        for i in range(max_id):
            if i not in current_ids:
                return f'cache_{i}'
        return f'cache_{max_id + 1}'
    
    def drop_cache(self, cache_id: str) -> None:
        """Drop an existing cache history.

        Args:
            cache_id (str): The cache ID of the file to drop.
        """
        cached_prompts = self.cached_prompts
        if cache_id in cached_prompts.keys():
            ps = cached_prompts.pop(cache_id)
            cache_file = os.path.join(self.cache_dir, f'{cache_id}.safetensors')
            if os.path.exists(cache_file):
                os.remove(cache_file)
                if self.logger is not None:
                    self.logger.info(f'Cache history "{cache_id}" dropped. Number of tokens: {ps.length}.')
        for k, v in cached_prompts.items():
            cached_prompts[k] = prompt_summary_to_dict(v)
        with open(self.json_dir, 'w') as f:
            json.dump(cached_prompts, f, indent=4)

    def find_cache(self, token_ids: List[int], token_threshold: Optional[int] = None) -> Tuple[Optional[CacheHistory], Optional[str]]:
        """Find a cache history given the new prompt token ids.

        Args:
            token_ids (List[int]): The new prompt token ids.
            token_threshold (Optional[int], optional): Minimum number of tokens in the cache history to be considered to use. If None is given, token_threshold of the cache manager will be used as default. Defaults to None.

        Returns:
            Tuple[Optional[CacheHistory], Optional[str]]: The cache history and it's cache ID.
        """
        cached_prompts = list(self.cached_prompts.items())
        if len(cached_prompts) == 0:
            return None, None
        token_threshold = self.token_threshold if token_threshold is None else token_threshold
        cached_prompts = [(x, find_max_prefix_num(new=token_ids, baseline=x[1].token_ids)) for x in cached_prompts]
        cached_prompts = list(filter(lambda x: x[1] >= token_threshold, cached_prompts))
        if len(cached_prompts) == 0:
            return None, None
        cached_prompts.sort(key=lambda x: x[1], reverse=True)
        if cached_prompts[0][1] == 0:
            return None, None
        key = cached_prompts[0][0][0]
        cache_file = os.path.join(self.cache_dir, f'{key}.safetensors')
        cache_history, metadata = load_cache(cache_file)
        self.update_cache_time(cache_id=key)
        self.logger.info(f'Cache history "{key}" with {cached_prompts[0][0][1].length} tokens is selected.')
        return cache_history, key
    
    def update_cache_time(self, cache_id: str) -> None:
        """Updating the last modified time of an existing cache history.

        Args:
            cache_id (str): The existing cache ID to update.
        """
        cached_prompts = self.cached_prompts
        for k, v in cached_prompts.items():
            cached_prompts[k] = prompt_summary_to_dict(v)
            if k == cache_id:
                cached_prompts[k]['last_modified'] = dt.now().timestamp()
        with open(self.json_dir, 'w') as f:
            json.dump(cached_prompts, f, indent=4)

    def save_cache(self, 
        cache: Union[StepOutput, CacheHistory],
        replace_threshold: float = 0.9,
        token_threshold: Optional[int] = None, 
        from_cache_id: Optional[str] = None, 
        metadata: Optional[Dict[str, str]] = None
    ) -> None:
        """Save a new cache history.

        Args:
            cache (Union[StepOutput, CacheHistory]): The new cache history to save.
            replace_threshold (float, optional): Percentage threshold of tokens prefix similar to the given from_cache_id to replace the given cache ID. Defaults to 0.9.
            token_threshold (Optional[int], optional): Minimum number of tokens in the cache history to be considered to be saved. If None is given, token_threshold of the cache manager will be used as default. Defaults to None.
            from_cache_id (Optional[str], optional): A cache ID that will be considered to be replaced (see replace_threshold). Defaults to None.
            metadata (Optional[Dict[str, str]], optional): Metadata to save in the cache history file. Defaults to None.
        """
        token_threshold = self.token_threshold if token_threshold is None else token_threshold
        new_len = len(cache.token_ids)
        if new_len < token_threshold:
            return
        cached_prompts = self.cached_prompts
        if from_cache_id in cached_prompts.keys():
            old_len = cached_prompts[from_cache_id].length
            max_prefix = find_max_prefix_num(new=cache.token_ids, baseline=cached_prompts[from_cache_id].token_ids)
            similarity = max_prefix / old_len
            if similarity > replace_threshold:
                if old_len > new_len:
                    self.update_cache_time(cache_id=from_cache_id)
                    return
                else:
                    self.drop_cache(cache_id=from_cache_id)
                    cached_prompts.pop(from_cache_id)
        
        duplicates = list(filter(lambda x: x[1].length == find_max_prefix_num(cache.token_ids, x[1].token_ids), cached_prompts.items()))
        for k, v in duplicates:
            self.drop_cache(cache_id=k)
            cached_prompts.pop(k)
                
        if len(cached_prompts) >= self.max_keep:
            exist_caches = list(cached_prompts.keys())
            cached_prompts = list(cached_prompts.items())
            cached_prompts.sort(key=lambda x: x[1].last_modified, reverse=True)
            cached_prompts = dict(cached_prompts[:self.max_keep - 1])
            new_caches = cached_prompts.keys()
            for cache_id in exist_caches:
                if cache_id not in new_caches:
                    self.drop_cache(cache_id=cache_id)

        new_id = self.new_cache_id
        ps = PromptSummary(token_ids=cache.token_ids, length=new_len, last_modified=dt.now())
        cached_prompts[new_id] = ps
        save_cache(cache=cache, filename=os.path.join(self.cache_dir, f'{new_id}.safetensors'), metadata=metadata)
        for k, v in cached_prompts.items():
            cached_prompts[k] = prompt_summary_to_dict(v)
        with open(self.json_dir, 'w') as f:
            json.dump(cached_prompts, f, indent=4)
        if self.logger is not None:
            self.logger.info(f'New cache history "{new_id}" saved. Number of tokens: {new_len}.')
        return
