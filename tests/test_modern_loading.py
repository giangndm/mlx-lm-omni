"""Regressions for loading Qwen audio checkpoints with modern MLX."""

import json

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten
from mlx_lm.utils import load_model

from mlx_lm_omni.models.qwen_omni.model import Model, ModelArgs
from mlx_lm_omni.tokenizer import ExtendedEmbedding
from mlx_lm_omni.utils import get_model_classes


def test_quantization_preserves_audio_embeddings():
    model = nn.Module()
    model.embed = ExtendedEmbedding(32, 64)
    nn.quantize(model, group_size=64, bits=4, mode="affine")
    audio = mx.full((1, 2, 64), 0.25)
    tokens = model.embed.embed_audio_chunk(audio)
    output = model.embed(mx.array([[1, *tokens, 2]]))
    mx.eval(output)
    np.testing.assert_array_equal(np.asarray(output[:, 1:3]), np.asarray(audio))
    assert not model.embed.extended_embedding_queue


@pytest.mark.parametrize("layout", ["pytorch", "mlx"])
def test_checkpoint_loads_audio_convolutions(tmp_path, layout):
    config = {
        "model_type": "qwen2_5_omni",
        "thinker_config": {
            "text_config": {
                "model_type": "qwen2",
                "hidden_size": 64,
                "num_hidden_layers": 1,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "rms_norm_eps": 1e-6,
                "vocab_size": 32,
            },
            "audio_config": {
                "num_mel_bins": 128,
                "d_model": 64,
                "n_window": 100,
                "encoder_attention_heads": 4,
                "encoder_layers": 1,
                "max_source_positions": 1500,
                "output_dim": 64,
            },
        },
    }
    model = Model(ModelArgs.from_dict(config))
    weights = dict(tree_flatten(model.parameters()))
    if layout == "pytorch":
        for name in ("conv1", "conv2"):
            key = f"thinker.audio_tower.{name}.weight"
            weights[key] = mx.swapaxes(weights[key], 1, 2)
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)

    loaded, _ = load_model(tmp_path, get_model_classes=get_model_classes, strict=True)
    output = loaded.thinker.audio_tower(mx.zeros((128, 10)))
    mx.eval(output)
    assert output.shape == (1, 2, 64)
