from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

_FILES = ["config.json", "model*.safetensors", "tokenizer.json"]


def resolve_model_dir(name_or_path: str | Path) -> Path:
    """A local directory as-is, otherwise a HF Hub repo id downloaded into the HF cache."""
    path = Path(name_or_path)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(name_or_path), allow_patterns=_FILES))


def load_state_dict(model_dir: str | Path) -> dict[str, torch.Tensor]:
    shards = sorted(Path(model_dir).glob("model*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no model*.safetensors in {model_dir}")
    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        state.update(load_file(shard))
    return state


def load_tokenizer(model_dir: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
