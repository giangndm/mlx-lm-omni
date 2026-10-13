import numpy as np
import mlx.nn as nn
import mlx.core as mx
from dataclasses import dataclass

from mlx_lm.tokenizer_utils import TokenizerWrapper, StreamingDetokenizer
from mlx_lm.utils import load_model

from mlx_lm_omni.audio_mel import AudioMel, AudioMelConfig
from mlx_lm_omni.model_path import get_model_path
from mlx_lm_omni.tokenizer import ExtendedEmbedding, ExtendedTokenizer, replace_slice

from .audio_tower import AudioTower, AudioTowerArgs
from .projector import Projector, ProjectorArgs
from .thinker import get_thinker_classes

AUDIO_SPECIAL_TOKEN = "<|reserved_special_token_0|>"

@dataclass
class ModelArgs:
    vocab_size: int
    hidden_size: int
    audio_config: AudioTowerArgs
    projector_config: ProjectorArgs
    text_model_id: str

    @staticmethod
    def from_dict(cfg: dict) -> "ModelArgs":
        cfg["audio_hidden_size"] = cfg["audio_config"]["d_model"]
        
        return ModelArgs(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            audio_config=AudioTowerArgs.from_dict(cfg["audio_config"]),
            projector_config=ProjectorArgs.from_dict(cfg),
            text_model_id=cfg["text_model_id"]
        )

class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        self.audio_tower = AudioTower(args.audio_config)
        self.multi_modal_projector = Projector(args.projector_config)
        
        text_model_path = get_model_path(args.text_model_id)
        text_model, _text_config = load_model(text_model_path, lazy=True, get_model_classes=get_thinker_classes)
        self.thinker = text_model

    @property
    def layers(self) -> list[nn.Module]:
        return self.thinker.layers
    
    def __call__(self, inputs: mx.array, cache = None) -> mx.array:
        return self.thinker(inputs, cache=cache)
    
    def build_custom_tokenizer(self, tokenizer: TokenizerWrapper) -> ExtendedTokenizer:
        return TokenizerWithAudio(self.audio_tower, self.multi_modal_projector, tokenizer, self.text_model.model.embed_tokens)

class TokenizerWithAudio(ExtendedTokenizer):
    def __init__(self, audio_tower: AudioTower, multi_modal_projector: Projector, tokenizer: TokenizerWrapper, embeddings: ExtendedEmbedding):
        self._audio_mel = AudioMel(AudioMelConfig.from_dict({
            "feature_size": 128,
            "sampling_rate": 16000,
            "hop_length": 160,
            "n_fft": 400,
        }))
        self._audio_tower = audio_tower
        self._multi_modal_projector = multi_modal_projector
        self._embeddings = embeddings
        self._tokenizer = tokenizer
        self._audio_special_token_id = self._tokenizer.encode(AUDIO_SPECIAL_TOKEN)[1:]
        
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
        mel = mx.expand_dims(mx.array(mel.T, dtype=mx.bfloat16), axis=0)
        audio_tower = self._audio_tower(mel)        
        features = self._multi_modal_projector(audio_tower)
        return self._embeddings.embed_audio_chunk(features)
    
    def apply_chat_template(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        # if the content is a list of audio, encode the audio
        for message in messages:
            if message.get("audio", None) is not None:
                extended_tokens = self.encode_audio(message["audio"])
                message["audio"] = extended_tokens
                if AUDIO_SPECIAL_TOKEN not in message["content"]:
                    message["content"] += AUDIO_SPECIAL_TOKEN
        
        tokens = self._tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)
        for message in messages:
            if message.get("audio", None) is not None:
                replace_slice(tokens, self._audio_special_token_id, message["audio"])
        return tokens
    
    def save_pretrained(self, path: str):
        self._tokenizer.save_pretrained(path)
