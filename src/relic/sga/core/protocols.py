"""
Defines protocols that the SGA API uses.
"""
from __future__ import annotations

from typing import (
    TypeVar,
    Protocol,
    BinaryIO,
    runtime_checkable,
)

T = TypeVar("T")


@runtime_checkable
class StreamSerializer(Protocol[T]):
    """Serializes the Type to/from a binary stream."""

    def unpack(self, stream: BinaryIO) -> T:
        """
        Converts binary data from the stream to parsed data.

        :param stream: The stream to read from.

        :return: The parsed data.
        """
        raise NotImplementedError

    def pack(self, stream: BinaryIO, value: T) -> int:
        """
        Converts binary data from the stream to parsed data.

        :param stream: The stream to write to.
        :param value: The data to convert to binary.

        :return: The number of bytes written.
        """
        raise NotImplementedError


__all__ = ["T", "StreamSerializer"]
