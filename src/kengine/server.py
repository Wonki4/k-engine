from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

from kengine.config import EngineConfig
from kengine.engine import Engine, StepOutput
from kengine.sequence import SamplingParams, SeqStatus, Sequence

logger = logging.getLogger(__name__)


class CompletionRequest(BaseModel):
    prompt: str
    model: str | None = None
    max_tokens: int = Field(default=64, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    seed: int | None = None
    stream: bool = False


class IncrementalDecoder:
    """Turns a growing token list into text deltas, holding back incomplete UTF-8 sequences."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.ids: list[int] = []
        self.emitted = ""

    def push(self, token_id: int) -> str:
        self.ids.append(token_id)
        text = self.tokenizer.decode(self.ids)
        if text.endswith("�"):
            return ""
        delta = text[len(self.emitted) :]
        self.emitted = text
        return delta


def create_app(engine: Engine, model_name: str = "kengine") -> FastAPI:
    # The engine is single-threaded. Handlers never touch it directly: they drop commands in
    # ``inbox`` and the loop task applies them between steps, on the event-loop thread.
    inbox: list[tuple[str, Sequence]] = []
    streams: dict[int, asyncio.Queue[StepOutput]] = {}
    wake = asyncio.Event()

    async def engine_loop() -> None:
        while True:
            while inbox:
                command, seq = inbox.pop(0)
                if command == "submit":
                    engine.submit(seq)
                elif seq.status is not SeqStatus.FINISHED:
                    engine.abort(seq)
            if not engine.has_work():
                wake.clear()
                await wake.wait()
                continue
            try:
                outputs = await asyncio.to_thread(engine.step)
            except Exception:
                # One bad step must not take the loop down with it: every later request
                # would hang forever. Fail what was in flight and keep serving.
                logger.exception("engine step failed; failing all in-flight requests")
                outputs = [StepOutput(seq.id, -1, "error") for seq in engine.abort_all()]
            for output in outputs:
                queue = streams.get(output.seq_id)
                if queue is not None:
                    queue.put_nowait(output)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(engine_loop())
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(title="kengine", lifespan=lifespan)

    async def run(seq: Sequence) -> AsyncIterator[tuple[str, str | None]]:
        """Queue ``seq`` and yield (text delta, finish reason) as tokens arrive."""
        queue: asyncio.Queue[StepOutput] = asyncio.Queue()
        streams[seq.id] = queue
        inbox.append(("submit", seq))
        wake.set()
        decoder = IncrementalDecoder(engine.tokenizer)
        try:
            while True:
                output = await queue.get()
                if output.finish_reason == "error":
                    yield "", "error"
                    return
                yield decoder.push(output.token_id), output.finish_reason
                if output.finish_reason is not None:
                    return
        finally:
            streams.pop(seq.id, None)
            if seq.status is not SeqStatus.FINISHED:  # client went away mid-generation
                inbox.append(("abort", seq))
                wake.set()

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        params = SamplingParams(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            seed=req.seed,
        )
        try:
            seq = engine.make_sequence(req.prompt, params)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        base = {
            "id": f"cmpl-{seq.id}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.model or model_name,
        }

        def chunk(text: str, finish_reason: str | None) -> dict:
            choice = {"index": 0, "text": text, "finish_reason": finish_reason}
            return {**base, "choices": [choice]}

        if req.stream:

            async def events() -> AsyncIterator[str]:
                async for text, finish_reason in run(seq):
                    if text or finish_reason:
                        yield f"data: {json.dumps(chunk(text, finish_reason))}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        parts: list[str] = []
        finish_reason = None
        async for text, finish_reason in run(seq):  # noqa: B007
            parts.append(text)
        if finish_reason == "error":
            raise HTTPException(status_code=500, detail="generation failed")
        body = chunk("".join(parts), finish_reason)
        prompt_tokens, completion_tokens = len(seq.prompt_ids), len(seq.output_ids)
        body["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return body

    @app.get("/v1/stats")
    async def stats() -> dict[str, int]:
        return engine.stats()

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve a Llama-family model with kengine.")
    parser.add_argument("--model", required=True, help="HF Hub repo id or local directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None, help="cpu, cuda, mps (default: auto)")
    parser.add_argument("--block-size", type=int, default=EngineConfig.block_size)
    parser.add_argument("--num-blocks", type=int, default=EngineConfig.num_blocks)
    parser.add_argument("--max-batch-size", type=int, default=EngineConfig.max_batch_size)
    args = parser.parse_args()

    engine = Engine.from_pretrained(
        args.model,
        EngineConfig(
            block_size=args.block_size,
            num_blocks=args.num_blocks,
            max_batch_size=args.max_batch_size,
            device=args.device,
        ),
    )
    uvicorn.run(create_app(engine, args.model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
