# adapted from mlx-lm
import json
from functools import partial
from typing import List, Optional

from transformers import AutoTokenizer, PreTrainedTokenizer
from huggingface_hub import hf_hub_download

REPLACEMENT_CHAR = "\ufffd"


def _remove_space(x):
    if x and x[0] == " ":
        return x[1:]
    return x


class NaiveDetokenizer:

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self._tokenizer = tokenizer
        self.reset()

    def reset(self, num_seqs: Optional[int] = None) -> None:
        self.num_seqs = num_seqs
        self.texts = [] if self.num_seqs is None else [''] * self.num_seqs
        self.tokens = [] if self.num_seqs is None else [[]] * self.num_seqs
        self.current_tokens = [] if self.num_seqs is None else [[]] * self.num_seqs
        self.final_segments = None

    def add_tokens(self, token_ids: List[List[int]]) -> None:
        if self.num_seqs is None:
            self.reset(num_seqs=len(token_ids))
        elif len(token_ids) != self.num_seqs:
            raise Exception('Number of tokens does not match with number of existing sequences.')
        self.current_tokens = [tids + nids for tids, nids in zip(self.current_tokens, token_ids)]

    def finalize(self) -> None:
        new_texts = self._tokenizer.batch_decode(self.current_tokens)
        new_texts = [nt.rstrip(REPLACEMENT_CHAR) for nt in new_texts]
        self.tokens = [t + ct for t, ct in zip(self.tokens, self.current_tokens)]
        self.texts = [t + nt for t, nt in zip(self.texts, new_texts)]
        self.current_tokens = [[]] * self.num_seqs
        self.final_segments = new_texts        

    @property
    def last_segments(self) -> List[str]:
        if self.final_segments is not None:
            return self.final_segments
        new_texts = self._tokenizer.batch_decode(self.current_tokens)
        with_repl = [newt.endswith(REPLACEMENT_CHAR) for newt in new_texts]
        bundle = [(ot if wr else ot + nt, nt if wr else [], os if wr else os + ns, '' if wr else ns) for ot, nt, os, ns, wr in zip(self.tokens, self.current_tokens, self.texts, new_texts, with_repl)]
        tokens, current_tokens, texts, new_segments = list(zip(*bundle))
        self.tokens = list(tokens)
        self.current_tokens = list(current_tokens)
        self.texts = list(texts)
        new_segments = list(new_segments)
        return new_segments

class SPMDetokenizer:
    """A streaming detokenizer for SPM models.

    It adds tokens to the text if the next token starts with the special SPM
    underscore which results in linear complexity.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, trim_space=True):
        self.trim_space = trim_space
        self.eos_token = tokenizer.eos_token

        # Extract the tokens in a list from id to text
        self.tokenmap = [""] * (max(tokenizer.vocab.values()) + 1)
        for value, tokenid in tokenizer.vocab.items():
            self.tokenmap[tokenid] = value

        # Replace bytes with their value
        for i in range(len(self.tokenmap)):
            if self.tokenmap[i].startswith("<0x"):
                self.tokenmap[i] = chr(int(self.tokenmap[i][3:5], 16))

        self.reset()

    def reset(self, num_seqs: Optional[int] = None) -> None:
        self.num_seqs = num_seqs
        self.offsets = [] if self.num_seqs is None else [0] * self.num_seqs
        self._unflushed = [] if self.num_seqs is None else [''] * self.num_seqs
        self.texts = [] if self.num_seqs is None else [''] * self.num_seqs
        self.tokens = [] if self.num_seqs is None else [[]] * self.num_seqs

    def add_tokens(self, token_ids: List[List[int]]) -> None:
        if self.num_seqs is None:
            self.reset(num_seqs=len(token_ids))
        elif len(token_ids) != self.num_seqs:
            raise Exception('Number of tokens does not match with number of existing sequences.')
        vs = [self.tokenmap[token[0]] for token in token_ids]
        for i, v in enumerate(vs):
            if v[0] == "\u2581":
                if self.texts[i] or not self.trim_space:
                    self.texts[i] += self._unflushed[i].replace("\u2581", " ")
                else:
                    self.texts[i] = _remove_space(self._unflushed[i].replace("\u2581", " "))
                self._unflushed[i] = v
            elif self.eos_token == v:
                if self.texts[i] or not self.trim_space:
                    self.texts[i] += self._unflushed[i].replace("\u2581", " ") + v
                else:
                    self.texts[i] = _remove_space(self._unflushed[i].replace("\u2581", " ")) + v
                self._unflushed[i] = ''
            else:
                self._unflushed[i] += v

    def finalize(self):
        for i in range(self.num_seqs):
            if self.texts[i] or not self.trim_space:
                self.texts[i] += self._unflushed[i].replace("\u2581", " ")
            else:
                self.texts[i] = _remove_space(self._unflushed[i].replace("\u2581", " "))
            self._unflushed[i] = ""

    @property
    def last_segments(self):
        """Return the last segment of readable text since last time this property was accessed."""
        texts = self.texts
        segments = []
        for i, text in enumerate(texts):
            if text and text[-1] != REPLACEMENT_CHAR:
                segments.append(text[self.offsets[i] :])
                self.offsets[i] = len(text)
            else:
                segments.append('')
        return segments



class BPEDetokenizer:
    """A streaming detokenizer for OpenAI style BPE models.

    It adds tokens to the text if the next token starts with a space similar to
    the SPM detokenizer.
    """

    _byte_decoder = None
    _space_matches = (".", "?", "!", ",", "'", "n't", "'m", "'s", "'ve", "'re")

    def __init__(self, tokenizer):

        self.clean_spaces = tokenizer.clean_up_tokenization_spaces

        # Extract the tokens in a list from id to text
        self.tokenmap = [None] * len(tokenizer.vocab)
        for value, tokenid in tokenizer.vocab.items():
            self.tokenmap[tokenid] = value

        self.reset()

        # Make the BPE byte decoder from
        # https://github.com/openai/gpt-2/blob/master/src/encoder.py
        self.make_byte_decoder()

    def reset(self, num_seqs: Optional[int] = None) -> None:
        self.num_seqs = num_seqs
        self.offsets = [] if self.num_seqs is None else [0] * self.num_seqs
        self._unflushed = [] if self.num_seqs is None else [''] * self.num_seqs
        self.texts = [] if self.num_seqs is None else [''] * self.num_seqs
        self.tokens = [] if self.num_seqs is None else [[]] * self.num_seqs

    def _maybe_trim_space(self, current_text):
        if len(current_text) == 0:
            return current_text
        elif current_text[0] != " ":
            return current_text
        elif not self.text:
            return current_text[1:]
        elif self.clean_spaces and current_text[1:].startswith(self._space_matches):
            return current_text[1:]
        return current_text

    def add_tokens(self, token_ids: List[List[int]]) -> None:
        if self.num_seqs is None:
            self.reset(num_seqs=len(token_ids))
        elif len(token_ids) != self.num_seqs:
            raise Exception('Number of tokens does not match with number of existing sequences.')
        vs = [self.tokenmap[token[0]] for token in token_ids]
        for i, v in enumerate(vs):
            if self._byte_decoder[v[0]] == 32:
                current_text = bytearray(
                    self._byte_decoder[c] for c in self._unflushed[i]
                ).decode("utf-8")
                self.texts[i] += self._maybe_trim_space(current_text)
                self._unflushed[i] = v
            else:
                self._unflushed[i] += v

    def finalize(self):
        for i in range(self.num_seqs):
            current_text = bytearray(self._byte_decoder[c] for c in self._unflushed[i]).decode(
                "utf-8"
            )
            self.texts[i] += self._maybe_trim_space(current_text)
            self._unflushed[i] = ""

    @property
    def last_segments(self):
        """Return the last segment of readable text since last time this property was accessed."""
        texts = self.texts
        segments = []
        for i, text in enumerate(texts):
            if text and text[-1] != REPLACEMENT_CHAR:
                segments.append(text[self.offsets[i] :])
                self.offsets[i] = len(text)
            else:
                segments.append('')
        return segments

    @classmethod
    def make_byte_decoder(cls):
        """See https://github.com/openai/gpt-2/blob/master/src/encoder.py for the rationale."""
        if cls._byte_decoder is not None:
            return

        char_to_bytes = {}
        limits = [
            0,
            ord("!"),
            ord("~") + 1,
            ord("¡"),
            ord("¬") + 1,
            ord("®"),
            ord("ÿ") + 1,
        ]
        n = 0
        for i, (start, stop) in enumerate(zip(limits, limits[1:])):
            if i % 2 == 0:
                for b in range(start, stop):
                    char_to_bytes[chr(2**8 + n)] = b
                    n += 1
            else:
                for b in range(start, stop):
                    char_to_bytes[chr(b)] = b
        cls._byte_decoder = char_to_bytes


class TokenizerWrapper:
    """A wrapper that combines an HF tokenizer and a detokenizer.

    Accessing any attribute other than the ``detokenizer`` is forwarded to the
    huggingface tokenizer.
    """

    def __init__(self, tokenizer, detokenizer_class=NaiveDetokenizer):
        self._tokenizer = tokenizer
        self._detokenizer_class = detokenizer_class
        self._detokenizer = detokenizer_class(tokenizer)

    def __getattr__(self, attr):
        if attr == "detokenizer":
            return self._detokenizer
        elif attr.startswith("_"):
            return self.__getattribute__(attr)
        else:
            return getattr(self._tokenizer, attr)

    def __setattr__(self, attr, value):
        if attr == "detokenizer":
            raise AttributeError("Cannot set the detokenizer.")
        elif attr.startswith("_"):
            super().__setattr__(attr, value)
        else:
            setattr(self._tokenizer, attr, value)



def _match(a, b):
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        return len(a) == len(b) and all(k in b and _match(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_match(ai, bi) for ai, bi in zip(a, b))

    return a == b


def _is_spm_decoder(decoder):
    _target_description = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0},
        ],
    }
    return _match(_target_description, decoder)


def _is_spm_decoder_no_space(decoder):
    _target_description = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
        ],
    }
    return _match(_target_description, decoder)


def _is_bpe_decoder(decoder):
    return isinstance(decoder, dict) and decoder.get("type", None) == "ByteLevel"


def load_tokenizer(model_path, tokenizer_config_extra={}):
    """Load a huggingface tokenizer and try to infer the type of streaming
    detokenizer to use.

    Note, to use a fast streaming tokenizer, pass a local file path rather than
    a Hugging Face repo ID.
    """
    detokenizer_class = NaiveDetokenizer

    tokenizer_file = model_path / "tokenizer.json"
    if not tokenizer_file.exists():
        tokenizer_file = hf_hub_download(repo_id=str(model_path), filename='tokenizer.json')
    with open(tokenizer_file, "r") as fid:
        tokenizer_content = json.load(fid)
    if "decoder" in tokenizer_content:
        if _is_spm_decoder(tokenizer_content["decoder"]):
            detokenizer_class = SPMDetokenizer
        elif _is_spm_decoder_no_space(tokenizer_content["decoder"]):
            detokenizer_class = partial(SPMDetokenizer, trim_space=False)
        elif _is_bpe_decoder(tokenizer_content["decoder"]):
            detokenizer_class = BPEDetokenizer

    return TokenizerWrapper(
        AutoTokenizer.from_pretrained(model_path, **tokenizer_config_extra),
        detokenizer_class,
    )