from __future__ import annotations

from datamodule.data_module import _batch_by_token_count


def test_max_frames_caps_padded_not_summed_workload():
    rows = [(0, 100), (1, 100), (2, 60), (3, 60), (4, 60)]
    batches = _batch_by_token_count(rows, max_frames=220)
    assert batches == [[0, 1], [2, 3, 4]]
    lengths = dict(rows)
    assert all(
        len(batch) * max(lengths[index] for index in batch) <= 220
        for batch in batches
    )


def test_batch_size_cap_and_singleton_are_respected():
    rows = [(0, 120), (1, 100), (2, 80)]
    batches = _batch_by_token_count(rows, max_frames=200, batch_size=2)
    assert batches == [[0], [1, 2]]
