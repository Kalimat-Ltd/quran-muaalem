import torch
import pytest

from quran_muaalem.decode import ctc_decode, CTCDecodeOut


@pytest.mark.parametrize(
    "batch_ids, batch_probs, ex_batch_ids, ex_batch_probs",
    [
        (
            [
                [1, 1, 0, 2, 2],
            ],
            [
                [0.3, 0.2, 0.9, 0.6, 0.8],
            ],
            [
                [1, 2],
            ],
            [
                [0.25, 0.7],
            ],
        ),
        # all diffrent
        (
            [
                [1, 2, 3, 4, 5],
            ],
            [
                [0.3, 0.2, 0.9, 0.6, 0.8],
            ],
            [
                [1, 2, 3, 4, 5],
            ],
            [
                [0.3, 0.2, 0.9, 0.6, 0.8],
            ],
        ),
        # all diffrent with blank
        (
            [
                [0, 0, 0, 1, 2, 3, 4, 5, 0, 0, 0],
            ],
            [
                [0.9, 0.9, 0.8, 0.3, 0.2, 0.9, 0.6, 0.8, 0.9, 0.9, 0.8],
            ],
            [
                [1, 2, 3, 4, 5],
            ],
            [
                [0.3, 0.2, 0.9, 0.6, 0.8],
            ],
        ),
        # all diffrent with blank in between
        (
            [
                [0, 0, 0, 1, 2, 0, 3, 4, 5, 0, 0, 0],
            ],
            [
                [0.9, 0.9, 0.8, 0.3, 0.2, 1.0, 0.9, 0.6, 0.8, 0.9, 0.9, 0.8],
            ],
            [
                [1, 2, 3, 4, 5],
            ],
            [
                [0.3, 0.2, 0.9, 0.6, 0.8],
            ],
        ),
        # all diffrent with blank in between
        (
            [
                [0, 0, 0, 1, 1, 1, 2, 0, 3, 4, 5, 0, 0, 0],
            ],
            [
                [0.9, 0.9, 0.8, 0.3, 0.2, 0.7, 0.8, 1.0, 0.9, 0.6, 0.8, 0.9, 0.9, 0.8],
            ],
            [
                [1, 2, 3, 4, 5],
            ],
            [
                [0.4, 0.8, 0.9, 0.6, 0.8],
            ],
        ),
        # complete example
        (
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2],
                [0, 0, 0, 1, 1, 1, 2, 0, 3, 4, 5, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.2, 0.9, 0.6, 0.8],
                [0.9, 0.9, 0.8, 0.3, 0.2, 0.7, 0.8, 1.0, 0.9, 0.6, 0.8, 0.9, 0.9, 0.8],
            ],
            [
                [1, 2],
                [1, 2, 3, 4, 5],
            ],
            [
                [0.25, 0.7],
                [0.4, 0.8, 0.9, 0.6, 0.8],
            ],
        ),
    ],
)
def test_ctc_decode(batch_ids, batch_probs, ex_batch_ids, ex_batch_probs):
    outs = ctc_decode(
        torch.LongTensor(batch_ids),
        torch.FloatTensor(batch_probs),
        collapse_consecutive=True,
    )
    for idx in range(len(batch_ids)):
        print(f"IDS: {outs[idx].ids}")
        print(f"EXP IDS: {torch.LongTensor(ex_batch_ids[idx])}")
        print(f"Probs: {outs[idx].p}")
        print(f"EXP Probs: {torch.FloatTensor(ex_batch_probs[idx])}")
        torch.testing.assert_close(outs[idx].ids, torch.LongTensor(ex_batch_ids[idx]))
        torch.testing.assert_close(outs[idx].p, torch.FloatTensor(ex_batch_probs[idx]))
