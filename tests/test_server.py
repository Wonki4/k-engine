from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from kengine.server import IncrementalDecoder, create_app
from tests.conftest import PROMPTS


@pytest.fixture
def client(make_engine):
    engine = make_engine()
    with TestClient(create_app(engine, "test-model")) as client:
        client.engine = engine
        yield client


def expected_text(tokenizer, hf_greedy, prompt: str, max_tokens: int) -> str:
    return tokenizer.decode(hf_greedy(tokenizer.encode(prompt).ids, max_tokens))


def test_completion_returns_reference_text(client, tokenizer, hf_greedy):
    response = client.post("/v1/completions", json={"prompt": PROMPTS[0], "max_tokens": 12})
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion" and body["model"] == "test-model"
    choice = body["choices"][0]
    # The model may hit EOS before 12 tokens; the reference ignores EOS, so compare a prefix.
    assert choice["text"] and expected_text(tokenizer, hf_greedy, PROMPTS[0], 12).startswith(
        choice["text"]
    )
    assert choice["finish_reason"] in ("length", "stop")
    usage = body["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_streaming_chunks_concatenate_to_the_full_completion(client):
    request = {"prompt": PROMPTS[1], "max_tokens": 16}
    full = client.post("/v1/completions", json=request).json()["choices"][0]

    with client.stream("POST", "/v1/completions", json={**request, "stream": True}) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in response.iter_lines() if line]

    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]
    assert len(chunks) > 1
    assert "".join(c["choices"][0]["text"] for c in chunks) == full["text"]
    assert [c["choices"][0]["finish_reason"] for c in chunks[:-1]] == [None] * (len(chunks) - 1)
    assert chunks[-1]["choices"][0]["finish_reason"] == full["finish_reason"]


def test_concurrent_requests_each_get_their_own_answer(client):
    def ask(prompt: str) -> str:
        body = client.post("/v1/completions", json={"prompt": prompt, "max_tokens": 16}).json()
        return body["choices"][0]["text"]

    solo = [ask(p) for p in PROMPTS]
    with ThreadPoolExecutor(max_workers=len(PROMPTS)) as pool:
        together = list(pool.map(ask, PROMPTS))
    assert together == solo


def test_stats_reports_prefix_reuse(client):
    request = {"prompt": PROMPTS[3], "max_tokens": 4}
    client.post("/v1/completions", json=request)
    client.post("/v1/completions", json=request)
    stats = client.get("/v1/stats").json()
    assert stats["prefix_hit_tokens"] > 0
    assert stats["running"] == stats["waiting"] == 0


def test_invalid_requests_are_rejected(client):
    assert client.post("/v1/completions", json={"prompt": ""}).status_code == 400
    too_long = {"prompt": PROMPTS[0], "max_tokens": 10**6}
    response = client.post("/v1/completions", json=too_long)
    assert response.status_code == 400 and "exceeds" in response.json()["detail"]
    assert client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 0}).status_code == 422


def test_incremental_decoder_holds_back_partial_characters(tokenizer):
    text = "서울은 한국의 수도 🙂 naïve"
    ids = tokenizer.encode(text).ids
    decoder = IncrementalDecoder(tokenizer)
    deltas = [decoder.push(i) for i in ids]
    assert "".join(deltas) == tokenizer.decode(ids)
    assert all("�" not in d for d in deltas)


def test_near_zero_temperature_is_served_as_greedy(client):
    request = {"prompt": PROMPTS[0], "max_tokens": 8}
    greedy = client.post("/v1/completions", json=request).json()["choices"][0]["text"]
    tiny = client.post("/v1/completions", json={**request, "temperature": 1e-300})
    assert tiny.status_code == 200
    assert tiny.json()["choices"][0]["text"] == greedy


def test_failed_step_fails_the_request_but_not_the_server(client):
    engine = client.engine
    real_step = engine.step

    def failing_step():
        engine.step = real_step
        raise RuntimeError("boom")

    engine.step = failing_step
    request = {"prompt": PROMPTS[0], "max_tokens": 8}
    assert client.post("/v1/completions", json=request).status_code == 500
    assert not engine.has_work()
    assert client.post("/v1/completions", json=request).status_code == 200
