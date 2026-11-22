from __future__ import annotations

import pytest
import torch

from kengine import Engine, EngineConfig, ModelConfig
from kengine.cache import BlockPool
from kengine.loader import load_tokenizer, resolve_model_dir
from kengine.model import Llama

MODEL = "HuggingFaceTB/SmolLM2-135M"

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village by the sea, there lived a fisherman who",
    "def fibonacci(n):\n    ",
    "Water boils at a temperature of 100 degrees Celsius at sea level. At higher altitudes, "
    "the boiling point drops because the air pressure is lower, which means that",
]


@pytest.fixture(scope="session")
def model_dir():
    return resolve_model_dir(MODEL)


@pytest.fixture(scope="session")
def model_cfg(model_dir) -> ModelConfig:
    return ModelConfig.from_file(model_dir / "config.json")


@pytest.fixture(scope="session")
def model(model_dir, model_cfg) -> Llama:
    return Llama.from_pretrained(model_dir, model_cfg, "cpu", torch.float32)


@pytest.fixture(scope="session")
def tokenizer(model_dir):
    return load_tokenizer(model_dir)


@pytest.fixture(scope="session")
def prompt_ids(tokenizer) -> list[list[int]]:
    return [tokenizer.encode(p).ids for p in PROMPTS]


@pytest.fixture(scope="session")
def hf_model(model_dir):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()


@pytest.fixture(scope="session")
def hf_greedy(hf_model):
    """Reference greedy decoding with no KV cache at all: a full forward pass per token."""
    memo: dict[tuple[tuple[int, ...], int], list[int]] = {}

    def run(ids: list[int], max_tokens: int) -> list[int]:
        key = (tuple(ids), max_tokens)
        if key not in memo:
            tokens = list(ids)
            with torch.inference_mode():
                for _ in range(max_tokens):
                    logits = hf_model(torch.tensor([tokens])).logits[0, -1]
                    tokens.append(int(logits.argmax()))
            memo[key] = tokens[len(ids) :]
        return memo[key]

    return run


@pytest.fixture
def make_pool(model_cfg):
    def make(num_blocks: int = 64, block_size: int = 16) -> BlockPool:
        return BlockPool(
            model_cfg.num_hidden_layers,
            num_blocks,
            block_size,
            model_cfg.num_key_value_heads,
            model_cfg.head_dim,
        )

    return make


@pytest.fixture
def make_engine(model, tokenizer, model_cfg):
    """Engines built here are checked for block leaks when the test ends."""
    engines: list[Engine] = []

    def make(**overrides) -> Engine:
        cfg = EngineConfig(**{"num_blocks": 64, "device": "cpu", **overrides})
        engine = Engine(model, tokenizer, model_cfg, cfg)
        engines.append(engine)
        return engine

    yield make

    for engine in engines:
        pool = engine.pool
        assert not engine.has_work()
        pool.check()
        # With every request finished, the prefix cache is the only thing holding blocks.
        assert pool.num_used == len(engine.prefix_cache)
        engine.prefix_cache.clear()
        assert pool.num_free == pool.num_blocks
