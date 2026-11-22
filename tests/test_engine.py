from __future__ import annotations

import pytest
import torch

from kengine import SamplingParams, SeqStatus
from kengine.sampler import sample

GREEDY = SamplingParams(max_tokens=40, ignore_eos=True)


def run_all(engine) -> None:
    while engine.has_work():
        engine.step()


@pytest.mark.parametrize("index", range(4))
def test_solo_greedy_matches_hf(make_engine, prompt_ids, hf_greedy, index):
    ids = prompt_ids[index]
    assert make_engine().generate([ids], GREEDY) == [hf_greedy(ids, 40)]


def test_batched_greedy_matches_hf(make_engine, prompt_ids, hf_greedy):
    engine = make_engine()
    outputs = engine.generate(prompt_ids, GREEDY)
    assert outputs == [hf_greedy(ids, 40) for ids in prompt_ids]
    assert engine.stats()["preemptions"] == 0


def test_requests_joining_mid_flight_match_hf(make_engine, prompt_ids, hf_greedy):
    """Continuous batching: new requests arrive while others are already decoding."""
    engine = make_engine()
    seqs = []
    for ids in prompt_ids:
        seqs.append(engine.add_request(ids, GREEDY))
        for _ in range(7):
            engine.step()
    run_all(engine)
    assert [s.output_ids for s in seqs] == [hf_greedy(ids, 40) for ids in prompt_ids]


def test_preempted_sequences_resume_with_identical_output(make_engine, prompt_ids, hf_greedy):
    # All four prompts are admitted (6 blocks), then outgrow the 8-block pool together.
    engine = make_engine(num_blocks=8)
    outputs = engine.generate(prompt_ids, GREEDY)
    assert engine.stats()["preemptions"] > 0
    assert outputs == [hf_greedy(ids, 40) for ids in prompt_ids]


def test_oldest_request_is_never_the_one_preempted(make_engine, prompt_ids):
    engine = make_engine(num_blocks=8)
    seqs = [engine.add_request(ids, GREEDY) for ids in prompt_ids]
    first = seqs[0]
    while first.status is not SeqStatus.FINISHED:
        engine.step()
        assert first.status is not SeqStatus.WAITING
    assert engine.stats()["preemptions"] > 0
    run_all(engine)


def test_max_batch_size_caps_concurrency(make_engine, prompt_ids, hf_greedy):
    engine = make_engine(max_batch_size=2)
    seqs = [engine.add_request(ids, GREEDY) for ids in prompt_ids]
    while engine.has_work():
        engine.step()
        assert len(engine.scheduler.running) <= 2
    assert [s.output_ids for s in seqs] == [hf_greedy(ids, 40) for ids in prompt_ids]


def test_repeated_prompt_reuses_cached_blocks(make_engine, prompt_ids, hf_greedy):
    engine = make_engine()
    ids = prompt_ids[3]  # 40+ tokens: at least two full blocks
    expected = hf_greedy(ids, 40)

    assert engine.generate([ids], GREEDY) == [expected]
    first_pass = engine.stats()
    assert first_pass["prefix_hit_tokens"] == 0

    assert engine.generate([ids], GREEDY) == [expected]
    second_pass = engine.stats()
    reusable = (len(ids) - 1) // 16 * 16
    assert second_pass["prefix_hit_tokens"] == reusable
    computed = second_pass["computed_tokens"] - first_pass["computed_tokens"]
    assert computed == first_pass["computed_tokens"] - reusable


def test_shared_prefix_with_different_suffix(make_engine, prompt_ids, hf_greedy):
    engine = make_engine()
    system = prompt_ids[3]
    a, b = system + prompt_ids[0], system + prompt_ids[2]
    assert engine.generate([a], GREEDY) == [hf_greedy(a, 40)]
    assert engine.generate([b], GREEDY) == [hf_greedy(b, 40)]
    assert engine.stats()["prefix_hit_tokens"] == len(system) // 16 * 16


def test_concurrent_request_reuses_prompt_of_one_still_running(make_engine, prompt_ids, hf_greedy):
    engine = make_engine()
    ids = prompt_ids[3]
    first = engine.add_request(ids, GREEDY)
    engine.step()  # prefill done, still generating
    second = engine.add_request(ids, GREEDY)
    run_all(engine)
    assert engine.stats()["prefix_hit_tokens"] > 0
    assert first.output_ids == second.output_ids == hf_greedy(ids, 40)


def test_prefix_cache_is_evicted_to_make_room(make_engine, prompt_ids, hf_greedy):
    engine = make_engine(num_blocks=6)
    for ids in prompt_ids:  # each run leaves blocks behind; later runs must reclaim them
        assert engine.generate([ids], GREEDY) == [hf_greedy(ids, 40)]
    assert engine.stats()["preemptions"] == 0


def test_stops_at_eos_and_at_stop_token(make_engine, prompt_ids, hf_greedy, model_cfg):
    ids = prompt_ids[0]
    reference = hf_greedy(ids, 40)
    engine = make_engine()

    seq = engine.add_request(ids, SamplingParams(max_tokens=40, stop_token_ids=(reference[5],)))
    run_all(engine)
    stop_at = reference.index(reference[5])
    assert seq.output_ids == reference[: stop_at + 1]
    assert seq.finish_reason == "stop"

    seq = engine.add_request(ids, SamplingParams(max_tokens=3, ignore_eos=True))
    run_all(engine)
    assert seq.output_ids == reference[:3] and seq.finish_reason == "length"


def test_abort_releases_blocks(make_engine, prompt_ids):
    engine = make_engine()
    running = engine.add_request(prompt_ids[1], GREEDY)
    engine.step()
    queued = engine.add_request(prompt_ids[2], GREEDY)
    engine.abort(running)
    engine.abort(queued)
    assert running.finish_reason == queued.finish_reason == "abort"
    assert not engine.has_work()


def test_rejects_requests_that_can_never_fit(make_engine, prompt_ids):
    engine = make_engine(num_blocks=2)  # 32 tokens of KV in total
    with pytest.raises(ValueError, match="KV pool capacity"):
        engine.add_request(prompt_ids[0], SamplingParams(max_tokens=64))
    with pytest.raises(ValueError, match="empty"):
        engine.add_request([], GREEDY)
    with pytest.raises(ValueError, match="context"):
        make_engine(num_blocks=1024).add_request(prompt_ids[0], SamplingParams(max_tokens=10**6))


def test_seeded_sampling_is_reproducible(make_engine, prompt_ids):
    params = SamplingParams(max_tokens=20, temperature=0.8, top_p=0.9, seed=1234, ignore_eos=True)
    engine = make_engine()
    first = engine.generate([prompt_ids[1]], params)
    second = engine.generate([prompt_ids[1]], params)
    other = engine.generate(
        [prompt_ids[1]],
        SamplingParams(max_tokens=20, temperature=0.8, top_p=0.9, seed=99, ignore_eos=True),
    )
    assert first == second
    assert first != other


# -- sampler ------------------------------------------------------------------------------------


def test_sampler_greedy_at_zero_temperature():
    logits = torch.tensor([0.1, 3.0, -1.0, 2.9])
    assert sample(logits, SamplingParams(temperature=0.0)) == 1


def test_sampler_top_k_restricts_candidates():
    logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    rng = torch.Generator().manual_seed(0)
    params = SamplingParams(temperature=5.0, top_k=2)
    assert {sample(logits, params, rng) for _ in range(200)} == {0, 1}


def test_sampler_top_p_keeps_smallest_covering_set():
    logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    rng = torch.Generator().manual_seed(0)
    drawn = {sample(logits, SamplingParams(temperature=1.0, top_p=0.7), rng) for _ in range(300)}
    assert drawn == {0, 1}
    # A tiny top_p still leaves the single most likely token.
    assert sample(logits, SamplingParams(temperature=1.0, top_p=0.01), rng) == 0


def test_sampler_treats_near_zero_temperature_as_greedy():
    logits = torch.tensor([0.1, 3.0, -1.0, 2.9])
    assert sample(logits, SamplingParams(temperature=1e-300)) == 1
