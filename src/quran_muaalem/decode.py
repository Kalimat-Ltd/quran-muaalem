from dataclasses import dataclass
import torch
import numpy as np
from numpy.typing import NDArray

from .modeling.vocab import PAD_TOKEN_IDX
from .muaalem_typing import Unit


@dataclass
class CTCDecodeOut:
    """
    Both are 1D Tensors
    """

    ids: torch.LongTensor
    p: torch.FloatTensor


def ctc_decode(
    batch_ids: torch.LongTensor,
    batch_probs: torch.FloatTensor,
    blank_id=PAD_TOKEN_IDX,
    collapse_consecutive=True,
) -> list[CTCDecodeOut]:
    """
    batch (torch.LongTensor): batch on integer ids of shape: batch, sequecne_len

    Return:
        list[tuple[list[int], float]]]:


    """
    outs = []
    assert batch_ids.shape == batch_probs.shape
    for seq_idx, seq in enumerate(batch_ids):
        if collapse_consecutive:
            tokens = []
            probs = []
            start = 0
            end = 0
            for idx in range(len(seq) - 1):
                curr = seq[idx]
                next = seq[idx + 1]
                # Last Item
                if idx == len(seq) - 2 and curr != blank_id:
                    if curr == next:
                        end = idx + 2
                        tokens.append(curr)
                        probs.append(
                            batch_probs[seq_idx][start:end].sum() / (end - start)
                        )
                    elif curr != next:
                        end = idx + 1
                        tokens.append(curr)
                        probs.append(
                            batch_probs[seq_idx][start:end].sum() / (end - start)
                        )
                        tokens.append(next)
                        probs.append(batch_probs[seq_idx][idx + 1])
                # Normal Case
                elif curr != next and curr != blank_id:
                    end = idx + 1
                    tokens.append(curr)
                    probs.append(batch_probs[seq_idx][start:end].sum() / (end - start))
                    start = end
                elif curr == blank_id:
                    start = idx + 1

            outs.append(
                CTCDecodeOut(
                    ids=torch.LongTensor(tokens),
                    p=torch.FloatTensor(probs),
                )
            )
        else:
            mask = seq != blank_id
            tokens = seq[mask]
            probs = batch_probs[seq_idx][mask]
            outs.append(CTCDecodeOut(ids=tokens, p=probs))
    return outs


def multilevel_greedy_decode(
    level_to_probs: dict[str, torch.FloatTensor],
) -> dict[str, Unit]: ...
