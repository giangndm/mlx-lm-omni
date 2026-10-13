import numpy as np
import mlx.nn as nn
import mlx.core as mx
from dataclasses import dataclass

from mlx_lm.tokenizer_utils import TokenizerWrapper, StreamingDetokenizer
from mlx_lm_omni.audio_mel import AudioMel, AudioMelConfig
from mlx_lm_omni.tokenizer import ExtendedEmbedding, ExtendedTokenizer, replace_slice

from .thinker import Thinker
from .audio_tower import AudioTower

AUDIO_SPECIAL_TOKEN = "<|AUDIO|>"

@dataclass
class ModelArgs:
    thinker_config: dict
    
    @staticmethod
    def from_dict(cfg: dict) -> "ModelArgs":
        cfg["thinker_config"]["text_config"]["model_type"] = "qwen2"
        return ModelArgs(
            thinker_config=cfg["thinker_config"]
        )

class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        self.thinker = Thinker(args.thinker_config)
        
    @property
    def layers(self) -> list[nn.Module]:
        return self.thinker.layers
    
    def __call__(self, inputs: mx.array, cache = None) -> mx.array:
        return self.thinker(inputs, cache=cache)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Convert checkpoint convolution layout before modern weight loading."""
        # New MLX weight loading no longer invokes the tower's update override.
        # Preserve its PyTorch-to-MLX conversion at the loader's sanitize boundary.
        for name in ("conv1", "conv2"):
            key = f"thinker.audio_tower.{name}.weight"
            if key not in weights:
                continue
            expected = getattr(self.thinker.audio_tower, name).weight.shape
            if weights[key].shape == (expected[0], expected[2], expected[1]):
                weights[key] = mx.swapaxes(weights[key], 1, 2)
            if weights[key].shape != expected:
                raise ValueError(
                    f"Unexpected convolution shape for {key}: {weights[key].shape}"
                )
        return weights

    def build_custom_tokenizer(self, tokenizer: TokenizerWrapper) -> ExtendedTokenizer:
        return TokenizerWithAudio(self.thinker.audio_tower, tokenizer, self.thinker.model.embed_tokens)

class TokenizerWithAudio(ExtendedTokenizer):
    def __init__(self, audio_tower: AudioTower, tokenizer: TokenizerWrapper, embeddings: ExtendedEmbedding):
        self._audio_mel = AudioMel(AudioMelConfig.from_dict({
            "feature_size": 128,
            "sampling_rate": 16000,
            "hop_length": 160,
            "n_fft": 400,
        }))
        self._audio_tower = audio_tower
        self._embeddings = embeddings
        self._tokenizer = tokenizer
        self._audio_special_token_id = self._tokenizer.encode(AUDIO_SPECIAL_TOKEN)
        
    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id
    
    @property
    def eos_token_ids(self) -> list[int]:
        return self._tokenizer.eos_token_ids
    
    @property
    def detokenizer(self) -> StreamingDetokenizer:
        return self._tokenizer.detokenizer
    
    def clean_up_tokenization_spaces(self) -> int:
        return self._tokenizer.clean_up_tokenization_spaces()

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)
    
    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)
        
    def encode_audio(self, audio: np.ndarray) -> list[int]:
        mel = self._audio_mel(audio)
        mel = mx.array(mel, dtype=mx.bfloat16)
        audio_tower = self._audio_tower(mel)
        return self._embeddings.embed_audio_chunk(audio_tower)
    
    def apply_chat_template(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        # if the content is a list of audio, encode the audio
        for message in messages:
            if message.get("audio", None) is not None:
                extended_tokens = self.encode_audio(message["audio"])
                message["audio"] = extended_tokens
                if AUDIO_SPECIAL_TOKEN not in message["content"]:
                    message["content"] += f"<|audio_bos|>{AUDIO_SPECIAL_TOKEN}<|audio_eos|>"
                elif f"<|audio_bos|>{AUDIO_SPECIAL_TOKEN}<|audio_eos|>" not in message["content"]:
                    message["content"] = message["content"].replace(AUDIO_SPECIAL_TOKEN, f"Audio 1: <|audio_bos|>{AUDIO_SPECIAL_TOKEN}<|audio_eos|>")
        
        tokens = self._tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)
        for message in messages:
            if message.get("audio", None) is not None:
                replace_slice(tokens, self._audio_special_token_id, message["audio"])
        return tokens
    
    def save_pretrained(self, path: str):
        self._tokenizer.save_pretrained(path)
