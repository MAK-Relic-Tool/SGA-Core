from __future__ import annotations

from fs.base import FS


class _EssenceFS(FS):
    ...


EssenceFS = _EssenceFS

__all__ = [
    "_EssenceFS",
]

