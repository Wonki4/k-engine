from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ModelConfig:
    """Llama-family decoder hyperparameters, parsed from a HF ``config.json``."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    max_position_embeddings: int = 2048
    eos_token_ids: tuple[int, ...] = ()
    bos_token_id: int | None = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_file(cls, path: str | Path) -> ModelConfig:
        raw = json.loads(Path(path).read_text())
        model_type = raw.get("model_type")
        if model_type != "llama":
            raise ValueError(f"unsupported model_type {model_type!r}; only 'llama' is supported")
        eos = raw.get("eos_token_id")
        if eos is None:
            eos_ids: tuple[int, ...] = ()
        elif isinstance(eos, int):
            eos_ids = (eos,)
        else:
            eos_ids = tuple(eos)
        heads = raw["num_attention_heads"]
        return cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=heads,
            num_key_value_heads=raw.get("num_key_value_heads") or heads,
            rms_norm_eps=raw.get("rms_norm_eps", 1e-5),
            rope_theta=raw.get("rope_theta", 10000.0),
            max_position_embeddings=raw.get("max_position_embeddings", 2048),
            eos_token_ids=eos_ids,
            bos_token_id=raw.get("bos_token_id"),
        )


@dataclass(frozen=True)
class EngineConfig:
    """Engine-side knobs. The KV pool holds ``num_blocks * block_size`` tokens in total."""

    block_size: int = 16
    num_blocks: int = 512
    max_batch_size: int = 32
    device: str | None = None
    dtype: torch.dtype | None = None

    def resolve_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def resolve_dtype(self) -> torch.dtype:
        if self.dtype is not None:
            return self.dtype
        return torch.float16 if self.resolve_device().type == "cuda" else torch.float32
