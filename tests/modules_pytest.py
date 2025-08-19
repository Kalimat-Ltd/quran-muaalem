import torch
import pytest

from quran_muaalem.decode import ctc_decode, CTCDecodeOut, multilevel_greedy_decode
from quran_muaalem.muaalem_typing import Unit


@pytest.mark.parametrize(
    "batch_ids, batch_probs, ex_batch_ids, ex_batch_probs",
    [
        (
            [
                [1],
            ],
            [
                [1.0],
            ],
            [
                [1],
            ],
            [
                [1.0],
            ],
        ),
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


@pytest.mark.parametrize(
    "level_to_probs, ex_level_to_units",
    [
        (
            {
                "phonemes": [
                    [
                        [0, 1, 0, 0, 0],
                    ],
                ]
            },
            {"phonemes": [Unit(text="a", probs=torch.FloatTensor([1.0]))]},
        ),
        # biger example
        (
            {
                "phonemes": [
                    [
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1],
                    ],
                ]
            },
            {
                "phonemes": [
                    Unit(text="abcd", probs=torch.FloatTensor([1.0, 1.0, 1.0, 1.0]))
                ]
            },
        ),
        # diffrent probs
        (
            {
                "phonemes": [
                    [
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0.6, 0.4, 0],
                        [0, 0, 0, 0, 1],
                    ],
                ]
            },
            {"phonemes": [Unit(text="abd", probs=torch.FloatTensor([1.0, 0.8, 1.0]))]},
        ),
        # diffrent probs and levels
        (
            {
                "phonemes": [
                    [
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0.6, 0.4, 0],
                        [0, 0, 0, 0, 1],
                    ],
                ],
                "hams": [
                    [
                        [0, 1, 0],
                        [0, 0.8, 0.2],
                    ],
                ],
            },
            {
                "phonemes": [
                    Unit(text="abd", probs=torch.FloatTensor([1.0, 0.8, 1.0]))
                ],
                "hams": [Unit(text="A", probs=torch.FloatTensor([0.9]))],
            },
        ),
        # diffrent probs and multiple sequences
        (
            {
                "phonemes": [
                    [
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0.6, 0.4, 0],
                        [0, 0, 0, 0, 1],
                    ],
                    [
                        [0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                    ],
                ],
                "hams": [
                    [
                        [0, 1, 0],
                        [0, 0.8, 0.2],
                    ],
                ],
            },
            {
                "phonemes": [
                    Unit(text="abd", probs=torch.FloatTensor([1.0, 0.8, 1.0])),
                    Unit(text="a", probs=torch.FloatTensor([1.0])),
                ],
                "hams": [Unit(text="A", probs=torch.FloatTensor([0.9]))],
            },
        ),
    ],
)
def test_multilevel_greedy_decode(level_to_probs, ex_level_to_units):
    level_to_vocab = {
        "phonemes": {1: "a", 2: "b", 3: "c", 4: "d"},
        "hams": {1: "A", 2: "B"},
    }
    level_to_probs = {l: torch.FloatTensor(p) for l, p in level_to_probs.items()}
    for level in level_to_probs:
        print(level_to_probs[level].shape)

    level_to_units = multilevel_greedy_decode(level_to_probs, level_to_vocab)
    assert len(ex_level_to_units) == len(level_to_units)
    assert set(ex_level_to_units.keys()) == set(level_to_units.keys())

    for level in ex_level_to_units:
        assert len(ex_level_to_units[level]) == len(level_to_units[level])
        for unit, ex_unit in zip(level_to_units[level], ex_level_to_units[level]):
            print(f"OUT UNIT: {unit}")
            print(f"EXP UNIT: {ex_unit}")
            assert unit.text == ex_unit.text
            torch.testing.assert_close(unit.probs, ex_unit.probs)
