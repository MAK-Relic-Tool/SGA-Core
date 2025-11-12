"""Definitions expressed concretely in core."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum, IntFlag
from functools import total_ordering
from typing import Any, Tuple, Iterable, Union, List, TypeVar

from relic.core.serialization import MagicWord

MAGIC_WORD = MagicWord(b"_ARCHIVE", name="SGA Magic Word")

# Safe os.O_FLAGS
# will fail for (value & flag) == flag; but will not interfere with (value | flag) use-cases

_T = TypeVar("_T")


def _has_get_attr(o: Any, name: str, default: _T) -> _T:
    if hasattr(o, name):
        return getattr(o, name)  # type: ignore
    return default


# Safe versions of existing flags
#  TODO; instead of working around the problem by using dummy flags; fix opens so that they are platform agnostic
# May already be mostly agnostic; docs say some constants are for unix/windows; but they may be lumping macos in with unix; meaning ONLY O_BINARY is the problem
class OSFlags(IntFlag):
    O_BINARY = _has_get_attr(os, "O_BINARY", 0)  # win only
    O_CREAT = _has_get_attr(os, "O_CREAT", 0)  # win/unix (macos?)
    O_RDONLY = _has_get_attr(os, "O_RDONLY", 0)  # win/unix (macos?)
    O_WRONLY = _has_get_attr(os, "O_WRONLY", 0)  # win/unix (macos?)
    O_TRUNC = _has_get_attr(os, "O_TRUNC", 0)  # win/unix (macos?)


@dataclass
@total_ordering
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

    def __len__(self) -> int:
        return 2

    def __getitem__(self, item: Union[int, slice]) -> Union[int, List[int]]:
        return self.as_tuple()[item]

    def as_tuple(self) -> Tuple[int, int]:
        return tuple(self)  # type: ignore

    def __eq__(self, other: object) -> bool:
        return self.as_tuple() == (
            other.as_tuple() if isinstance(other, Version) else other
        )

    def __lt__(self, other: Any) -> bool:
        cmp: bool = self.as_tuple() < (
            other.as_tuple() if isinstance(other, Version) else other
        )
        return cmp

    def __hash__(self) -> int:
        return self.as_tuple().__hash__()


class StorageType(int, Enum):
    """Specifies whether data is stored as a 'raw blob' or as a 'zlib compressed
    blob'."""

    # According to modpackager
    STORE = 0
    STREAM_COMPRESS = 1
    BUFFER_COMPRESS = 2


class VerificationType(
    int, Enum
):  # TODO; consider not sharing this; this is format specific and wasn't introduced until V4? It could be reimplemented in each version; since each version may have different values
    """A 'Flag' used to specify how the data's Redundancy Check is stored."""

    NONE = 0  # unknown real values, assuming incremental
    CRC = 1  # unknown real values, assuming incremental
    CRC_BLOCKS = 2  # unknown real values, assuming incremental
    MD5_BLOCKS = 3  # unknown real values, assuming incremental
    SHA1_BLOCKS = 4  # unknown real values, assuming incremental


__all__ = ["MAGIC_WORD", "Version", "StorageType", "VerificationType", "OSFlags"]
