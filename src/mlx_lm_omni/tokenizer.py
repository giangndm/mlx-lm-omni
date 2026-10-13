from abc import ABC, abstractmethod
import mlx.core as mx
import logging
import numpy as np
import math
from mlx.nn.layers.base import Module
from mlx_lm.tokenizer_utils import StreamingDetokenizer

EXTENDED_EMBEDDING_THRESHOLD = 5000000

class ExtendedTokenizer(ABC):
    @property
    @abstractmethod
    def detokenizer(self) -> StreamingDetokenizer:
        pass
    
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        pass
    
    @abstractmethod
    def encode_audio(self, audio: np.ndarray) -> list[int]:
        pass

class ExtendedEmbedding(Module):
    """Implements a simple lookup table that maps each input integer to a
    high-dimensional vector.

    Typically used to embed discrete tokens for processing by neural networks.

    Args:
        num_embeddings (int): How many possible discrete tokens can we embed.
           Usually called the vocabulary size.
        dims (int): The dimensionality of the embeddings.
    """

    def __init__(self, num_embeddings: int, dims: int):
        super().__init__()
        scale = math.sqrt(1 / dims)
        self.weight = mx.random.normal(shape=(num_embeddings, dims), scale=scale)
        self.extended_embedding_seed = EXTENDED_EMBEDDING_THRESHOLD
        self.extended_embedding_queue = []

    def _extra_repr(self):
        return f"{self.weight.shape[0]}, {self.weight.shape[1]}"

    def __call__(self, inputs: mx.array) -> mx.array:
        # split into chunks with value < 5000000 and > 5000000
        embeddings = []
        chunk_normal = inputs[0][0] < EXTENDED_EMBEDDING_THRESHOLD
        chunk_begin = 0
        
        def add_chunk(begin, end):
            if chunk_normal:
                embeddings.append(self.weight[inputs[:, begin:end]])
            else:
                extended_embedding = self.extended_embedding_queue.pop(0)
                logging.info(f"[ExtendedEmbeddings] pop extended embeddings: {inputs[0][begin]} len {extended_embedding.shape[1]} vs {end - begin}")
                embeddings.append(extended_embedding)
        
        for i in range(1, inputs.shape[1]):
            normal_token = inputs[0][i] < EXTENDED_EMBEDDING_THRESHOLD
            if chunk_normal != normal_token:
                add_chunk(chunk_begin, i)
                chunk_normal = not chunk_normal
                chunk_begin = i
            elif not chunk_normal and i > 0 and inputs[0][i] != inputs[0][i-1]: # multi continuous extended embeddings
                add_chunk(chunk_begin, i)
                chunk_begin = i

        # add last chunk
        add_chunk(chunk_begin, inputs.shape[1])
        
        if len(embeddings) > 1:
            return mx.concatenate(embeddings, axis=1)
        else:
            return embeddings[0]

    def as_linear(self, x):
        """
        Call the embedding layer as a linear layer.

        Use this for example when input embedding and output projection
        weights are tied.
        """
        return x @ self.weight.T

    def to_quantized(self, group_size: int = 64, bits: int = 4, mode: str = "affine"):
        """Return a :obj:`QuantizedEmbedding` layer that approximates this embedding layer."""
        if mode != "affine":
            raise ValueError("Extended embeddings support only affine quantization")
        return ExtendedQuantizedEmbedding.from_embedding(self, group_size, bits)
    
    def embed_audio_chunk(self, audio_embeddings: mx.array) -> list[int]:
        """
        Extend the embeddings to the original shape
        """
        extended_tokens = [self.extended_embedding_seed] * audio_embeddings.shape[1]
        self.extended_embedding_seed += 1
        self.extended_embedding_queue.append(audio_embeddings)
        return extended_tokens
    
class ExtendedQuantizedEmbedding(Module):
    """The same as :obj:`Embedding` but with a  quantized weight matrix.

    :obj:`QuantizedEmbedding` also provides a :meth:`from_embedding`
    classmethod to convert embedding layers to :obj:`QuantizedEmbedding`
    layers.

    Args:
        num_embeddings (int): How many possible discrete tokens can we embed.
           Usually called the vocabulary size.
        dims (int): The dimensionality of the embeddings.
        group_size (int, optional): The group size to use for the quantized
            weight. See :func:`~mlx.core.quantize`. Default: ``64``.
        bits (int, optional): The bit width to use for the quantized weight.
            See :func:`~mlx.core.quantize`. Default: ``4``.
    """

    def __init__(
        self,
        num_embeddings: int,
        dims: int,
        group_size: int = 64,
        bits: int = 4,
    ):
        super().__init__()

        # Quantization config
        self.group_size = group_size
        self.bits = bits

        # Initialize the quantized weight
        scale = math.sqrt(1 / dims)
        weight = mx.random.normal(shape=(num_embeddings, dims), scale=scale)
        self.weight, self.scales, self.biases = mx.quantize(weight, group_size, bits)
        self.num_embeddings = num_embeddings
        self.dims = dims
        
        self.extended_embedding_seed = EXTENDED_EMBEDDING_THRESHOLD
        self.extended_embedding_queue = []

        # Freeze this model's parameters
        self.freeze()

    def __call__(self, inputs: mx.array):
        # split into chunks with value < 5000000 and > 5000000
        embeddings = []
        chunk_normal = inputs[0][0] < EXTENDED_EMBEDDING_THRESHOLD
        chunk_begin = 0
        
        def add_chunk(begin, end):
            if chunk_normal:
                x = inputs[:, begin:end]
                embeddings.append(mx.dequantize(
                    self["weight"][x],
                    scales=self["scales"][x],
                    biases=self["biases"][x],
                    group_size=self.group_size,
                    bits=self.bits,
                ))
            else:
                extended_embedding = self.extended_embedding_queue.pop(0)
                logging.info(f"[ExtendedQuantizedEmbedding] pop extended embeddings: {inputs[0][begin]} len {extended_embedding.shape[1]} vs {end - begin}")
                embeddings.append(extended_embedding)
        
        for i in range(1, inputs.shape[1]):
            normal_token = inputs[0][i] < EXTENDED_EMBEDDING_THRESHOLD
            if chunk_normal != normal_token:
                add_chunk(chunk_begin, i)
                chunk_normal = not chunk_normal
                chunk_begin = i
            elif not chunk_normal and i > 0 and inputs[0][i] != inputs[0][i-1]: # multi continuous extended embeddings
                add_chunk(chunk_begin, i)
                chunk_begin = i

        # add last chunk
        add_chunk(chunk_begin, inputs.shape[1])
        
        if len(embeddings) > 1:
            return mx.concatenate(embeddings, axis=1)
        else:
            return embeddings[0]

    def as_linear(self, x):
        """
        Call the quantized embedding layer as a quantized linear layer.

        Use this for example when input embedding and output projection
        weights are tied.
        """
        return mx.quantized_matmul(
            x,
            self["weight"],
            scales=self["scales"],
            biases=self["biases"],
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )

    def _extra_repr(self):
        return (
            f"{self.num_embeddings}, {self.dims}, "
            f"group_size={self.group_size}, bits={self.bits}"
        )

    @classmethod
    def from_embedding(
        cls, embedding_layer: Module, group_size: int = 64, bits: int = 4
    ):
        """Create a :obj:`QuantizedEmbedding` layer from an :obj:`Embedding` layer."""
        embedding_dims, dims = embedding_layer.weight.shape
        ql = cls(embedding_dims, dims, group_size, bits)
        ql.weight, ql.scales, ql.biases = mx.quantize(
            embedding_layer.weight, group_size, bits
        )
        return ql
    
    def embed_audio_chunk(self, audio_embeddings: mx.array) -> list[int]:
        """
        Extend the embeddings to the original shape
        """
        extended_tokens = [self.extended_embedding_seed] * audio_embeddings.shape[1]
        self.extended_embedding_seed += 1
        self.extended_embedding_queue.append(audio_embeddings)
        return extended_tokens
    
def replace_slice(lst, search, target):
    """
    Replaces the first occurrence of a sublist `search` in `lst` with the sublist `target`.

    Args:
        lst (list): The list to modify.
        search (list): The slice to find and replace.
        target (list): The replacement slice.

    Returns:
        bool: True if replacement was made, False if search not found.
    """
    n = len(search)
    if n == 0:
        return False

    for i in range(len(lst) - n + 1):
        if lst[i:i + n] == search:
            lst[i:i + n] = target
            return True

    return False
