from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "Muaalem",
    "MuaalemOutput",
    "Unit",
    "Sifa",
    "SingleUnit",
    "explain_for_terminal",
]


if TYPE_CHECKING:  # pragma: no cover - imported only for type checkers
    from .inference import Muaalem  # noqa: F401
    from .muaalem_typing import MuaalemOutput, Unit, Sifa, SingleUnit  # noqa: F401
    from .explain import explain_for_terminal  # noqa: F401


_LAZY_IMPORTS = {
    "Muaalem": "quran_muaalem.inference",
    "MuaalemOutput": "quran_muaalem.muaalem_typing",
    "Unit": "quran_muaalem.muaalem_typing",
    "Sifa": "quran_muaalem.muaalem_typing",
    "SingleUnit": "quran_muaalem.muaalem_typing",
    "explain_for_terminal": "quran_muaalem.explain",
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'quran_muaalem' has no attribute {name!r}")
    module = import_module(_LAZY_IMPORTS[name])
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
