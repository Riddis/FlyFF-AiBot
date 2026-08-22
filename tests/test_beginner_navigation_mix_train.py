"""2026-08-14 Beginner Navigation Training Mix, Part 2 deferred amendment:
deterministic test proving make_stream_rngs's per-worker, per-stream RNG
separation actually holds -- identical (train_seed_base, continuation_seed,
worker_rank) tuples must reproduce identical initial draw sequences, and
different tuples (different worker_rank OR different continuation_seed)
must diverge. Also checks the four named streams within one worker are
themselves mutually distinct.
"""
from __future__ import annotations

from simulator.scratchpad.scratchpad_beginner_navigation_mix_train import TRAIN_SEED_BASE, make_stream_rngs


def _draws(rngs: dict, n: int = 5) -> dict:
    return {name: [float(rng.uniform()) for _ in range(n)] for name, rng in rngs.items()}


class TestStreamRngSeparation:
    def test_identical_tuple_reproduces_identical_draws(self):
        a = _draws(make_stream_rngs(TRAIN_SEED_BASE, 100, 0))
        b = _draws(make_stream_rngs(TRAIN_SEED_BASE, 100, 0))
        assert a == b

    def test_different_worker_rank_diverges(self):
        a = _draws(make_stream_rngs(TRAIN_SEED_BASE, 100, 0))
        b = _draws(make_stream_rngs(TRAIN_SEED_BASE, 100, 1))
        assert a != b
        for name in a:
            assert a[name] != b[name], f"stream {name!r} did not diverge across worker_rank"

    def test_different_continuation_seed_diverges(self):
        a = _draws(make_stream_rngs(TRAIN_SEED_BASE, 100, 0))
        b = _draws(make_stream_rngs(TRAIN_SEED_BASE, 102, 0))
        assert a != b
        for name in a:
            assert a[name] != b[name], f"stream {name!r} did not diverge across continuation_seed"

    def test_streams_within_one_worker_are_mutually_distinct(self):
        draws = _draws(make_stream_rngs(TRAIN_SEED_BASE, 100, 0))
        values = list(draws.values())
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                assert values[i] != values[j], (
                    f"streams {list(draws.keys())[i]!r} and {list(draws.keys())[j]!r} produced "
                    f"identical draws -- not actually independent"
                )

    def test_all_three_continuation_seeds_and_all_four_workers_pairwise_distinct(self):
        from simulator.scratchpad.scratchpad_beginner_navigation_mix_train import CONTINUATION_SEEDS, N_ENVS

        all_draws = []
        keys = []
        for seed in CONTINUATION_SEEDS:
            for rank in range(N_ENVS):
                all_draws.append(_draws(make_stream_rngs(TRAIN_SEED_BASE, seed, rank)))
                keys.append((seed, rank))
        for i in range(len(all_draws)):
            for j in range(i + 1, len(all_draws)):
                assert all_draws[i] != all_draws[j], f"{keys[i]} and {keys[j]} produced identical draws"
