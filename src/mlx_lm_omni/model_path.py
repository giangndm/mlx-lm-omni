# Copyright © 2023-2024 Apple Inc.
# Adapted from mlx-lm 0.24.1; see LICENSE.mlx-lm for the MIT license.

"""Resolve model snapshots without relying on removed MLX-LM internals."""

from pathlib import Path

from huggingface_hub import snapshot_download


def get_model_path(path_or_hf_repo: str, revision: str | None = None) -> Path:
    """Use a local model path or download the same files as the former loader."""
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                path_or_hf_repo,
                revision=revision,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.py",
                    "tokenizer.model",
                    "*.tiktoken",
                    "tiktoken.model",
                    "*.txt",
                    "*.jsonl",
                ],
            )
        )
    return model_path
