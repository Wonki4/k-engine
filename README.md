# k-engine

KV 캐시 관리를 중심에 둔 LLM 추론 엔진. 텐서 연산은 PyTorch에 맡기고, 모델 실행·캐시·스케줄링·서빙은 직접 구현한다. Llama 계열 디코더를 지원한다.

## 실행

```bash
uv sync
uv run python -m kengine.server --model HuggingFaceTB/SmolLM2-135M

curl -N localhost:8000/v1/completions -H 'content-type: application/json' \
  -d '{"prompt": "Hello", "max_tokens": 32, "stream": true}'
curl localhost:8000/v1/stats   # 블록 사용량, 프리픽스 적중, 선점 횟수
```

라이브러리로 쓸 때:

```python
from kengine import Engine, EngineConfig, SamplingParams

engine = Engine.from_pretrained("HuggingFaceTB/SmolLM2-135M", EngineConfig(num_blocks=256))
outputs = engine.generate(["The capital of France is"], SamplingParams(max_tokens=32))
print(engine.tokenizer.decode(outputs[0]))
```

## 구조

```
요청 → server → scheduler → model → sampler → 스트림
                    │          │
                    └── cache (블록 풀 · 블록 테이블 · 프리픽스 캐시)
```

| 모듈 | 역할 |
|---|---|
| `cache/block_pool.py` | 시작 시 한 번 할당하는 KV 텐서. free list와 참조 카운트로 블록 단위 대여 |
| `cache/block_table.py` | 시퀀스의 논리 위치 → 물리 블록 매핑. `fork()`와 copy-on-write |
| `cache/prefix_cache.py` | 토큰 해시 체인으로 꽉 찬 블록을 요청 간 재사용. LRU 축출 |
| `model/` | 디코더 forward. K/V를 모듈이 아니라 풀에 쓰고, 블록 테이블로 gather해서 어텐션 |
| `scheduler.py` | FCFS. 블록이 모자라면 가장 최근 시퀀스를 선점 |
| `engine.py` | `step()` 한 번이 스케줄 → forward → 샘플. 매 스텝 배치를 다시 짜는 연속 배칭 |
| `server.py` | OpenAI 호환 `/v1/completions` (SSE 스트리밍) |

시퀀스의 진행 상태는 "K/V가 풀에 들어 있는 토큰 수" 하나뿐이다. 매 스텝 그 뒤의 토큰을 전부 넣으므로 prefill, decode, 선점 후 재개가 같은 코드 경로를 탄다.

## 테스트

```bash
uv run pytest
uv run ruff check
```

정답 기준은 `transformers`로 로딩한 같은 모델이다. logits는 1e-4 이내로, greedy 생성은 단독·배치·중간 합류·선점 후 재개·프리픽스 적중 모두에서 토큰 단위로 일치해야 한다. 엔진을 쓰는 모든 테스트는 종료 시 블록 누수가 없는지 확인한다.
