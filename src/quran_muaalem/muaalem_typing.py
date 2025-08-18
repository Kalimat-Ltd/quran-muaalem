from typing import Sequence
from dataclasses import dataclass


@dataclass
class Unit:
    text: str
    probs: Sequence[float] | float


@dataclass
class SingleUnit:
    text: str
    prob: float


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
