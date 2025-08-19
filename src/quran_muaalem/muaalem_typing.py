from typing import Sequence
from dataclasses import dataclass
import torch


@dataclass
class Unit:
    """
    probs: 1D tensors
    """

    text: str
    probs: torch.FloatTensor | list[float]
    ids: torch.LongTensor | list[int]


@dataclass
class SingleUnit:
    text: str
    prob: float
    idx: int


@dataclass
class Sifa:
    phonemes_group: str
    hams_or_jahr: SingleUnit | None
    shidda_or_rakhawa: SingleUnit | None
    tafkheem_or_taqeeq: SingleUnit | None
    itbaq: SingleUnit | None
    safeer: SingleUnit | None
    qalqla: SingleUnit | None
    tikraar: SingleUnit | None
    tafashie: SingleUnit | None
    istitala: SingleUnit | None
    ghonna: SingleUnit | None


@dataclass
class MuaalemOutput:
    phonemes: Unit
    sifat: list[Sifa]
