"""
Definitions expressed concretely in core
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Tuple, Iterable

MagicWord = "_ARCHIVE".encode("ascii")


@dataclass
class Version:
    """A Version object.

    Args:
        major (int): The Major Version; Relic refers to this as the 'Version'.
        minor (int): The Minor Version; Relic refers to this as the 'Product'.
    """

    major: int
    minor: int = 0

    def __str__(self) -> str:
        return f"Version {self.major}.{self.minor}"

    def __iter__(self) -> Iterable[int]:
        yield self.major
        yield self.minor

    def __getitem__(self, item) -> int:
        if 0 <= item < 2:
            return self.major if item is 0 else self.minor
        else:
            raise KeyError(f"Index out of range, {item} not in [0,1]")

    def as_tuple(self) -> Tuple[int, int]:
        return tuple(self)  # type: ignore

    def __eq__(self, other: object) -> bool:
        return self.as_tuple() == (
            other.as_tuple() if isinstance(other, Version) else other
        )

    def __ne__(self, other: object) -> bool:
        return self.as_tuple() != (
            other.as_tuple() if isinstance(other, Version) else other
        )

    def __lt__(self, other: Any) -> bool:
        return self.as_tuple() < (
            other.as_tuple() if isinstance(other, Version) else other
        )

    def __gt__(self, other: Any) -> bool:
        return self.as_tuple() > (
            other.as_tuple() if isinstance(other, Version) else other
        )

    def __le__(self, other: Any) -> bool:
        return self.as_tuple() <= (
            other.as_tuple() if isinstance(other, Version) else other
        )

    def __ge__(self, other: Any) -> bool:
        return self.as_tuple() >= (
            other.as_tuple() if isinstance(other, Version) else other
        )

    def __hash__(self) -> int:
        return self.as_tuple().__hash__()


class StorageType(int, Enum):
    """
    Specifies whether data is stored as a 'raw blob' or as a 'zlib compressed blob'
    """

    # According to modpackager
    STORE = 0
    STREAM_COMPRESS = 1
    BUFFER_COMPRESS = 2


class VerificationType(
    int, Enum
):  # TODO; consider not sharing this; this is format specific and wasn't introduced until V4? It could be reimplemented in each version; since each version may have different values
    """
    A 'Flag' used to specify how the data's Redundancy Check is stored.
    """

    NONE = 0  # unknown real values, assuming incremental
    CRC = 1  # unknown real values, assuming incremental
    CRC_BLOCKS = 2  # unknown real values, assuming incremental
    MD5_BLOCKS = 3  # unknown real values, assuming incremental
    SHA1_BLOCKS = 4  # unknown real values, assuming incremental


__all__ = ["MagicWord", "Version", "StorageType", "VerificationType"]
