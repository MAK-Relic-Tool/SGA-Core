from __future__ import annotations

from abc import ABC
from typing import Tuple, Iterator

from fs.base import FS


class EssenceFS(FS, ABC):
    ...

    def iterate_fs(self) -> Iterator[Tuple[str, FS]]:
        raise NotImplementedError


__all__ = [
    "EssenceFS",
]
