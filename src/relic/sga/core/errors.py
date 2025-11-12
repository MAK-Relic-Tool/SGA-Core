"""Error definitions for the SGA API."""

from typing import List, Optional, Generic, TypeVar

from relic.core.errors import MismatchError, RelicToolError

from relic.sga.core.definitions import Version

_T = TypeVar("_T")


class MagicMismatchError(MismatchError[bytes]):
    """The archive did not specify the correct magic word."""

    def __init__(self, received: Optional[bytes], expected: Optional[bytes] = None):
        super().__init__("Magic Word", received, expected)


class VersionMismatchError(MismatchError[Version]):
    """A version did not match the version expected."""

    def __init__(
        self, received: Optional[Version] = None, expected: Optional[Version] = None
    ):
        super().__init__("Version", received, expected)


class VersionNotSupportedError(RelicToolError):
    """An unknown version was provided."""

    def __init__(self, received: Version, allowed: List[Version]):
        super().__init__()
        self.received = received
        self.allowed = allowed

    def __str__(self) -> str:
        def str_ver(version: Version) -> str:  # dont use str(version); too verbose
            return f"{version.major}.{version.minor}"

        allowed_str = [str_ver(_) for _ in self.allowed]
        return f"Version `{str_ver(self.received)}` is not supported. Versions supported: `{allowed_str}`"


class DecompressedSizeMismatch(MismatchError[int]):
    """A file was decompressed, but did not match the expected size."""

    def __init__(self, received: Optional[int] = None, expected: Optional[int] = None):
        super().__init__("Decompressed Size", received, expected)


class HashMismatchError(MismatchError[_T], Generic[_T]):
    """A sentinel class for catching all hash mismatch errors."""


class Md5MismatchError(HashMismatchError[bytes]): ...


class Crc32MismatchError(HashMismatchError[int]): ...


class Sha1MismatchError(HashMismatchError[bytes]):  #
    ...


__all__ = [
    "VersionMismatchError",
    "VersionNotSupportedError",
    "DecompressedSizeMismatch",
    "HashMismatchError",
    "Md5MismatchError",
    "Crc32MismatchError",
    "Sha1MismatchError",
    "MagicMismatchError",
]
