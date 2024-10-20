from .cache_utils import CacheHistory, save_cache, load_cache, find_max_prefix_num, split_cache, trim_cache
from mlx_lm.models.cache import KVCache
import mlx.core as mx
import json, os
from logging import Logger
from copy import deepcopy
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
    def __init__(self, cache_dir: str, token_threshold: int = 20, save_similarity_threshold: float = 0.9, max_keep: int = 50, logger: Optional[Logger] = None) -> None:
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
        self.save_sim_threshold = save_similarity_threshold
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
    
    def _get_new_cache_id(self, exist_ids: List[str]) -> str:
        current_ids = [int(x.removeprefix('cache_')) for x in exist_ids]
        if len(current_ids) == 0:
            return 'cache_0'
        max_id = max(current_ids)
        for i in range(max_id):
            if i not in current_ids:
                return f'cache_{i}'
        return f'cache_{max_id + 1}'

    
    def _log_info(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)
    
    def drop_cache(self, cache_id: Union[str, List[str]]) -> None:
        """Drop an existing cache history.

        Args:
            cache_id (Union[str, List[str]]): The cache ID or list of cache IDs to drop.
        """
        cached_prompts = self.cached_prompts
        cache_id = [cache_id] if isinstance(cache_id, str) else list(set(cache_id))
        for cid in cache_id:
            if cid in cached_prompts.keys():
                ps = cached_prompts.pop(cid)
                cache_file = os.path.join(self.cache_dir, f'{cid}.safetensors')
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    self._log_info(f'Cache history "{cid}" dropped. Number of tokens: {ps.length}.')
        for k, v in cached_prompts.items():
            cached_prompts[k] = prompt_summary_to_dict(v)
        with open(self.json_dir, 'w') as f:
            json.dump(cached_prompts, f, indent=4)

    def find_cache(self, token_ids: Union[List[int], List[List[int]]], token_threshold: Optional[int] = None) -> Tuple[Optional[CacheHistory], Optional[List[str]]]:
        """Find a cache history given the new prompt token ids.

        Args:
            token_ids (List[int]): The new prompt token ids.
            token_threshold (Optional[int], optional): Minimum number of tokens in the cache history to be considered to use. If None is given, token_threshold of the cache manager will be used as default. Defaults to None.

        Returns:
            Tuple[Optional[CacheHistory], Optional[str]]: The cache history and it's cache ID.
        """
        # Make them list of list of tokens
        token_ids = [token_ids] if isinstance(token_ids[0], int) else token_ids
        prompt_lens = [len(tids) for tids in token_ids]
        max_prompt_len = max(prompt_lens)

        # Calculate the total number of prompts
        num_prompts = len(token_ids)
        self._log_info(f'Number of prompts: {num_prompts}')

        # Make the prompt tokens tuples to find unique sets for querying, avoiding repetitive search for large batches
        token_tuples = [tuple(tids) for tids in token_ids]
        prompt_dict = dict(zip(range(num_prompts), token_tuples))
        find_queries = list(set(token_tuples))
        
        # Get existing prompt caches for search
        cached_prompts = self.cached_prompts

        # Early exit if no existing prompts
        if len(cached_prompts) == 0:
            self._log_info('No existing cache.')
            return None, None
        # Get the number of token token threshold
        token_threshold = self.token_threshold if token_threshold is None else token_threshold

        # Searching query by query
        found_cache = dict()
        for query in find_queries:
            q_len = len(query)
            res = [(
                k, # cache id
                find_max_prefix_num(query, v.token_ids), # number of tokens to get
                q_len - len(v.token_ids), # inverse number of tokens in the cache, easier to sort
                max_prompt_len - q_len, # prompt offset due to padding
                ) for k, v in cached_prompts.items()] # Get the highest number of shared prefix with smallest cache
            res.sort(key=lambda x: x[1:3], reverse=True)
            found_cache[query] = res[0] if res[0][1] >= token_threshold else (None, 0, 0, 0)

        prompt_dict = {k: found_cache[v] for k, v in prompt_dict.items()}
        used_cache = [v[0] for v in found_cache.values() if v[0] is not None]
        used_cache = list(set(used_cache))

        if len(used_cache) == 0:
            self._log_info(f'No cache with prefix match tokens larger than {token_threshold}.')
            return None, None

        # Load the caches
        cache_list = []
        for cid in used_cache:
            cache, metadata = load_cache(os.path.join(self.cache_dir, f'{cid}.safetensors'))
            cache_list.append((cid, cache))

        # Calculate the max position of the cache
        max_cache_len = min([v[3] + v[1] for v in found_cache.values()])
        max_cache_len = max_cache_len - 1 if max_cache_len == max_prompt_len else max_cache_len

        # Making empty keys and values pairs of tensors from found cache
        sample_cache = cache_list[0][1]
        kv_list = []
        for layer in sample_cache:
            _n, num_kv_heads, _nt, head_dim = layer.state[0].shape
            kv_list.append(
                (
                    mx.zeros((num_prompts, num_kv_heads, max_cache_len, head_dim), dtype=layer.state[0].dtype),
                    mx.zeros((num_prompts, num_kv_heads, max_cache_len, head_dim), dtype=layer.state[1].dtype)
                )
            )
        cache_dict = dict(cache_list)
        
        # filling the tensors with old cache
        new_cache = []
        for i, (keys, values) in enumerate(kv_list):
            new_cache.append(KVCache())
            for p_index, p_ids in prompt_dict.items():
                cid, max_prefix, _, offset = p_ids
                if max_cache_len - offset > 0:
                    keys[p_index:p_index + 1, :, offset:, :] = cache_dict[cid][i].state[0][:, :, :max_cache_len - offset, :]
                    values[p_index:p_index + 1, :, offset:, :] = cache_dict[cid][i].state[1][:, :, :max_cache_len - offset, :]      
            new_cache[i].update_and_fetch(keys, values)
            mx.eval(new_cache[i].state)
        self.update_cache_time(used_cache)
        uc_str = '", "'.join(used_cache)
        self._log_info(f'Using existing cache "{uc_str}" for {max_cache_len} tokens for each prompt.')
        del cache_list, cache_dict
        mx.metal.clear_cache()
        return CacheHistory(cache=new_cache, token_ids=[]), used_cache

    def update_cache_time(self, cache_id: Union[str, List[str]]) -> None:
        """Updating the last modified time of an existing cache history.

        Args:
            cache_id (str): The existing cache ID to update.
        """
        cache_id = [cache_id] if isinstance(cache_id, str) else cache_id
        cached_prompts = self.cached_prompts
        for k, v in cached_prompts.items():
            cached_prompts[k] = prompt_summary_to_dict(v)
            if k in cache_id:
                cached_prompts[k]['last_modified'] = dt.now().timestamp()
        with open(self.json_dir, 'w') as f:
            json.dump(cached_prompts, f, indent=4)

    def save_cache(self, 
        cache: CacheHistory,
        token_threshold: Optional[int] = None, 
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
        save_threshold = self.save_sim_threshold
        token_ids = [tuple(tids) for tids in cache.token_ids]
        prompt_lens = [len(tids) for tids in cache.token_ids]
        max_lens = max(prompt_lens)
        offsets = [max_lens - pl for pl in prompt_lens]
        caches = split_cache(cache=cache.cache)
        unique_caches_tokens = []
        unique_caches = []
        for i, tids in enumerate(token_ids):
            if (tids not in unique_caches_tokens) and prompt_lens[i] >= token_threshold:
                unique_caches_tokens.append(tids)
                unique_caches.append((tids, prompt_lens[i], offsets[i], trim_cache(caches[i], offset=offsets[i])))
        del caches, cache
        mx.metal.clear_cache()
        if len(unique_caches) == 0:
            self._log_info(f'No new caches saved as they are shorter than {token_threshold} tokens.')
            return
        
        # Compare among the unique caches and see if they have similar ones
        to_drop = []
        for tids, plen, offset, cache in unique_caches:
            if tids in unique_caches_tokens:
                comp_list = list(filter(lambda x: x[0] != tids, unique_caches))
                for comp in comp_list:
                    max_pref = find_max_prefix_num(tids, comp[0])
                    osim = max_pref / plen
                    csim = max_pref / comp[1]
                    max_sim = max(osim, csim)
                    if max_sim > save_threshold:
                        drop_tids = tids if osim == max_sim else comp[0]
                        to_drop.append(drop_tids)
        unique_caches = list(filter(lambda x: x[0] not in to_drop, unique_caches))

        # Comparing with existing caches
        exist_to_drop = []
        exist_to_keep = []
        cached_prompts = self.cached_prompts
        for tids, plen, offset, cache in unique_caches:
            if tids in unique_caches_tokens:
                for cid, ps in cached_prompts.items():
                    max_pref = find_max_prefix_num(tids, ps.token_ids)
                    osim = max_pref / plen
                    csim = max_pref / ps.length
                    max_sim = max(osim, csim)
                    if max_sim > save_threshold:
                        if osim == max_sim:
                            to_drop.append(tids)
                            exist_to_keep.append(cid)
                        else:
                            exist_to_drop.append(cid)
        unique_caches = list(filter(lambda x: x[0] not in to_drop, unique_caches)) 
        self.drop_cache(exist_to_drop)   

        # Final number of prompts to save
        num_save = len(unique_caches)
        if num_save > 0:
            cached_prompts = self.cached_prompts
            exist_to_drop = None
            if len(cached_prompts) + num_save > self.max_keep:
                num_to_drop = len(cached_prompts) + num_save - self.max_keep
                cp_ordered = list(cached_prompts.items())
                cp_ordered.sort(key=lambda x: x[1].last_modified)
                cp_ordered = list(filter(lambda x: x[0] not in exist_to_keep, cp_ordered))
                exist_to_drop = [x[0] for x in cp_ordered[:num_to_drop]]
            if exist_to_drop:
                self.drop_cache(exist_to_drop)
            lm = dt.now()
            cached_prompts = self.cached_prompts
            exist_ids = list(cached_prompts.keys())
            for tids, length, offset, che in unique_caches:
                new_id = self._get_new_cache_id(exist_ids=exist_ids)
                cached_prompts[new_id] = PromptSummary(token_ids=list(tids), length=length, last_modified=lm)
                exist_ids.append(new_id)
                save_cache(che, file=os.path.join(self.cache_dir, f'{new_id}.safetensors'))
                self._log_info(f'New cache "{new_id}" saved. Prompt length: {length}.')
            cached_prompts = {k: prompt_summary_to_dict(v) for k, v in cached_prompts.items()}
            with open(self.json_dir, 'w') as f:
                json.dump(cached_prompts, f, indent=4)
        else:
            self._log_info(f'New caches are not saved as old ones covered them already.')
        return
