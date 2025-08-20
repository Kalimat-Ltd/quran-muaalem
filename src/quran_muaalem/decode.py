from typing import Sequence
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
    batch_ids (torch.LongTensor): batch on integer ids of shape: batch, sequecne_len
    batch_probs (torch.LongTensor): batch on float32 ids of shape: batch, sequecne_len

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
            if len(seq) == 1 and seq[0] != blank_id:
                tokens.append(seq[0])
                probs.append(batch_probs[seq_idx][0])

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
    level_to_id_to_vocab: dict[str, dict[int, str]],
) -> dict[str, list[Unit]]:
    level_to_units = {}
    for level in level_to_probs:
        batch_probs, batch_ids = level_to_probs[level].topk(1, dim=-1)
        decode_outs = ctc_decode(
            batch_ids.squeeze(-1), batch_probs.squeeze(-1), collapse_consecutive=True
        )
        level_to_units[level] = []
        for decode_out in decode_outs:
            text = ""
            for idx in decode_out.ids:
                text += level_to_id_to_vocab[level][int(idx)]
            level_to_units[level].append(
                Unit(text=text, probs=decode_out.p, ids=decode_out.ids)
            )

    return level_to_units


def align_sequence(
    seq: Sequence[int] | torch.LongTensor, target_len: int, min_repeat: int = 3
) -> list[int]:
    """Aligns a sequence by removing items from the longest repateted items

    Returns:
        list[int]: the ids which are goning to be deleted if longest_repeat > len(seq) - target_len

    Example:
                seq = [1, 0, 1, 0, 0, 0, 0, 1], target_len = 7
                                ^  ^  ^
    Longest Repeat              ^  ^  ^
    Ouput: [3]
    """

    if len(seq) <= target_len:
        return []

    longest_start = 0
    longest_repeat = 0
    curr_repeat = 1
    curr_start = 0
    for idx in range(len(seq) - 1):
        curr = seq[idx]
        next = seq[idx + 1]
        if curr == next:
            curr_repeat += 1
        if (curr != next) or (idx == len(seq) - 2):
            if curr_repeat > longest_repeat and curr_repeat >= min_repeat:
                longest_repeat = curr_repeat
                longest_start = curr_start
            curr_start = idx + 1
            curr_repeat = 1

    # logical case to remote only from the longest repeat
    if longest_repeat > len(seq) - target_len:
        return list(range(longest_start, longest_start + len(seq) - target_len))
    else:
        return []
