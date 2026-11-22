# k-engine

An LLM inference engine built around KV cache management.

Tensor math is left to PyTorch. Everything above it is implemented here: the decoder forward pass, the KV cache, scheduling, sampling, and an OpenAI-compatible server. The engine, not the model, owns all KV memory, which is what makes paging, prefix sharing, copy-on-write, and preemption possible.

This is a from-scratch engine written to understand how serving systems such as vLLM work inside. It is small enough to read end to end (about 1,200 lines).

## Status

| | |
|---|---|
| Models | Llama-family decoders (`model_type: llama`) in Hugging Face safetensors format |
| Verified on | `HuggingFaceTB/SmolLM2-135M`, CPU, fp32 (macOS) |
| Not yet run | CUDA, fp16, MPS, any other checkpoint |
| Not measured | Throughput, latency, memory use |

Correctness is checked against `transformers`: logits match within 1e-4, and greedy output is token-for-token identical across solo, batched, mid-flight join, preempt-and-resume, and prefix-cache-hit runs.

See [Limitations](#limitations) before using this for anything beyond experimentation.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m kengine.server --model HuggingFaceTB/SmolLM2-135M
```

The model is downloaded from the Hugging Face Hub on first run. A local directory containing `config.json`, `model*.safetensors`, and `tokenizer.json` also works.

```bash
# Streaming
curl -N localhost:8000/v1/completions -H 'content-type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 32, "stream": true}'

# Non-streaming
curl localhost:8000/v1/completions -H 'content-type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 32}'

# Cache and scheduler counters
curl localhost:8000/v1/stats
```

### As a library

```python
from kengine import Engine, EngineConfig, SamplingParams

engine = Engine.from_pretrained("HuggingFaceTB/SmolLM2-135M", EngineConfig(num_blocks=256))

outputs = engine.generate(
    ["The capital of France is", "Once upon a time"],
    SamplingParams(max_tokens=32),
)
for ids in outputs:
    print(engine.tokenizer.decode(ids))
```

To drive the engine yourself, submit requests and call `step()`:

```python
seq = engine.add_request("Hello", SamplingParams(max_tokens=16, temperature=0.8, seed=0))
while engine.has_work():
    for out in engine.step():          # one sampled token per running sequence
        print(out.seq_id, out.token_id, out.finish_reason)
```

Requests can be added between steps; they join the running batch on the next one.

## Server

### Command-line options

| Flag | Default | Meaning |
|---|---|---|
| `--model` | required | Hub repo id or local directory |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8000` | Port |
| `--device` | auto | `cpu`, `cuda`, or `mps`. Auto picks `cuda` when available, else `cpu` |
| `--block-size` | `16` | Tokens per KV block |
| `--num-blocks` | `512` | Blocks in the KV pool. Total KV capacity is `block-size × num-blocks` tokens |
| `--max-batch-size` | `32` | Maximum sequences running at once |

The KV pool is allocated once at startup. Its size in bytes is:

```
num_layers × 2 × num_blocks × block_size × num_kv_heads × head_dim × bytes_per_value
```

For SmolLM2-135M in fp32 the default pool (8,192 tokens) is 360 MB.

### `POST /v1/completions`

| Field | Type | Default | Notes |
|---|---|---|---|
| `prompt` | string | required | Must be non-empty |
| `max_tokens` | int | `64` | At least 1 |
| `temperature` | float | `0.0` | `0` is greedy. Values below `1e-5` are treated as greedy |
| `top_p` | float | `1.0` | In `(0, 1]` |
| `top_k` | int | `0` | `0` disables |
| `seed` | int | none | Makes sampling reproducible |
| `stream` | bool | `false` | Server-sent events, ending with `data: [DONE]` |
| `model` | string | none | Echoed in the response. Not used for routing |

Responses follow the OpenAI text completion shape. Non-streaming responses include `usage`.

`finish_reason` is one of:

| Value | Meaning |
|---|---|
| `stop` | The model produced an end-of-sequence token |
| `length` | `max_tokens` was reached |
| `error` | The engine failed mid-generation (streaming only; non-streaming returns HTTP 500) |

A request is rejected with HTTP 400 when `prompt + max_tokens` exceeds either the model's context length or the total KV pool capacity. A request that could never fit would otherwise wait forever.

### `GET /v1/stats`

| Field | Meaning |
|---|---|
| `blocks_total`, `blocks_free` | KV pool size and free blocks |
| `blocks_prefix_cached` | Blocks held by the prefix cache |
| `running`, `waiting` | Sequences in each scheduler queue |
| `computed_tokens` | Tokens run through the model since startup |
| `prefix_hit_tokens` | Tokens skipped because their KV was already cached |
| `preemptions` | Times a sequence was evicted to free memory |

## Architecture

```
request → server → scheduler → model → sampler → stream
                       │          │
                       └── cache (block pool · block tables · prefix cache)
```

| Module | Responsibility |
|---|---|
| `config.py` | `ModelConfig` parsed from `config.json`; `EngineConfig` for engine knobs |
| `loader.py` | Hub download, safetensors loading, tokenizer |
| `cache/block_pool.py` | The single preallocated KV tensor; free list and reference counts |
| `cache/block_table.py` | Per-sequence map from token position to physical block; fork and copy-on-write |
| `cache/prefix_cache.py` | Cross-request reuse of full blocks, keyed by a token hash chain; LRU eviction |
| `model/layers.py` | RMSNorm, SwiGLU MLP, rotary position embedding |
| `model/attention.py` | `PagedAttention`: writes K/V to the pool, reads context back through the block table |
| `model/llama.py` | Decoder forward pass and weight loading |
| `sampler.py` | Greedy, temperature, top-k, top-p |
| `sequence.py` | A request and the blocks it holds |
| `scheduler.py` | Admission, preemption, queue ordering |
| `engine.py` | `step()`: schedule → build batch → forward → sample |
| `server.py` | HTTP API and the engine loop |

### KV memory: blocks and slots

All KV memory is one tensor, allocated at startup:

```
[num_layers, 2 (K/V), num_blocks, block_size, num_kv_heads, head_dim]
```

A **block** holds the keys and values of `block_size` consecutive tokens for every layer. A **slot** is one token's place in the pool: `block_id × block_size + offset`. The same slot index is valid in every layer, so a sequence needs only one block list.

Blocks are reference counted. A block returns to the free list when its count reaches zero. Several owners can hold the same block: two forked sequences, or a sequence and the prefix cache.

### Block tables

Each sequence has a block table: a list of block ids plus the number of tokens whose K/V are stored.

```
position:   0 ........ 15 | 16 ........ 31 | 32 .. 37
blocks:     [    7     ]   [     2      ]   [  11   ]     num_tokens = 38
```

- `reserve(n)` allocates whatever blocks are missing for `n` more tokens. It either succeeds completely or leaves the table untouched.
- `fork()` creates a second table sharing every block. Nothing is copied.
- **Copy-on-write:** if the next write would land in a partly filled block that another owner also holds, `reserve` first replaces it with a private copy. Full blocks are never written again, so they stay shared.

### Paged attention

The model holds no cache state. For each layer, attention:

1. Computes Q, K, V for the new tokens and applies rotary embedding to Q and K.
2. Writes the new K and V into the pool at the slots reserved for them.
3. Reads the sequence's whole context back by gathering its slots, then runs standard scaled dot-product attention with a causal mask.

Rotary positions are baked into the stored keys, so cached keys never need recomputing.

One forward pass is described by a `PagedBatch`. Each sequence in it may have a different number of cached tokens and a different number of new tokens; shorter rows are padded and masked.

The gather in step 3 is the portable baseline. The `KVStore` protocol in `model/attention.py` is the boundary where a kernel that attends over blocks in place would plug in.

### One code path for prefill, decode, and resume

A sequence's only progress marker is `num_cached`: how many of its tokens have K/V in the pool. Every step feeds `tokens[num_cached:]`:

| Situation | `num_cached` | Tokens fed |
|---|---|---|
| Just admitted | 0, or the prefix cache hit | The rest of the prompt |
| Decoding | all but the last | The one token sampled last step |
| Resumed after preemption | 0, or whatever is still cached | Everything not cached |

There is no separate prefill or recovery logic. Within a step, sequences feeding many tokens and sequences feeding one run as two batches, so a long prompt does not pad every decoding sequence to its length.

### Prefix cache

The K/V of a token depend on every token before it, so a block can be reused only by a request with the identical prefix. Blocks are keyed by a hash chain:

```
key_0 = H(tokens of block 0)
key_i = H(key_{i-1}, tokens of block i)
```

- Only full blocks are cached.
- On admission, the scheduler looks up the longest cached prefix and starts the sequence with those blocks already in its table. The last prompt token is always left to compute, because its logits choose the first output token.
- Blocks are offered to the cache when a prompt finishes prefilling, when a sequence finishes, and when it is preempted.
- The cache holds one reference per block. A block is evictable only when that is the last reference.
- Eviction is least-recently-used, and within one chain the deepest block goes first. Evicting a shallow block first would strand its descendants, since lookup walks the chain from the start.
- When the pool runs out of free blocks, it asks the prefix cache to evict before failing.

### Scheduling and preemption

Scheduling is first come, first served, with KV blocks as the budget.

Each step:

1. **Running sequences**, oldest first, reserve blocks for the tokens they are about to feed.
2. If a reservation fails, the **newest** running sequence is preempted: its blocks are released and it returns to the front of the waiting queue. The step retries until the reservation succeeds.
3. **Waiting sequences** are admitted in order while blocks and batch slots last. Running sequences are never preempted to admit a new one.

The oldest sequence is never the victim while others run, so some request always makes progress. A preempted sequence's full blocks remain in the prefix cache until evicted, so resuming is often cheaper than starting over.

Because the batch is rebuilt every step, requests join and leave between steps. This is continuous batching.

### Server concurrency

The engine is single-threaded. HTTP handlers never touch it: they place commands in an inbox, and one loop task applies them between steps. Each `step()` runs in a worker thread so the event loop stays responsive.

If a step raises, the loop logs it, fails every in-flight request, releases their blocks, and keeps serving.

## Testing

```bash
uv run pytest        # downloads SmolLM2-135M on first run; about a minute on a laptop CPU
uv run ruff check
```

| File | What it covers |
|---|---|
| `tests/test_cache.py` | Block pool, block table, and prefix cache as pure data structures, with tiny synthetic pools |
| `tests/test_model.py` | Logits against `transformers`: prefill, token-by-token decoding, ragged batches, forked sequences |
| `tests/test_engine.py` | Greedy output against a cache-free reference under batching, mid-flight joins, preemption, prefix reuse, eviction; stop conditions; sampling |
| `tests/test_server.py` | HTTP API: streaming, concurrency, validation, failure recovery |

The greedy reference deliberately uses no KV cache at all: a full `transformers` forward pass per token. Every engine built in a test is checked for leaked blocks when the test ends.

## Limitations

**Verification**
- Only one checkpoint on CPU in fp32 has been run. The CUDA and fp16 paths exist but are untested.
- No performance numbers exist.

**Cache and attention**
- Every layer copies the whole context out of the pool on every step. Cost grows with context length.
- Slot indices are built in a Python loop each step.
- Requests admitted in the same step do not share a common prefix with each other; each computes it.
- Partly filled blocks are not cached, so shared prefixes shorter than one block are not reused.
- `fork()` exists in the cache layer but is not exposed through the engine or API.

**Scheduling**
- A long prompt is prefilled in one pass, delaying other sequences' tokens for that step.
- If the request at the head of the queue does not fit, smaller requests behind it wait.
- No priorities, fairness, or per-request timeouts.

**Server**
- No authentication, rate limiting, or bound on queue length.
- The prefix cache is shared by all callers. Response timing and `/v1/stats` can reveal whether another caller sent a prompt with the same prefix. Do not expose this server to mutually untrusted users.
- Only `/v1/completions`. No chat endpoint, chat templates, tool calls, stop strings, logprobs, or multiple choices.

**Scope**
- One model per process, one device.
- Llama-family architecture only. No mixture-of-experts, quantized weights, or image input.

## Roadmap

- Run and benchmark on CUDA.
- A paged attention kernel that reads blocks in place instead of gathering.
- A model protocol and registry so other architectures can be added.
- Multiple models in one server, routed by the `model` field.
- Per-caller prefix cache isolation alongside authentication.

## License

Apache License 2.0. See [LICENSE](LICENSE).
