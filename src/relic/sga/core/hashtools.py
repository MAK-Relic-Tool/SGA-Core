import hashlib
import zlib
from typing import BinaryIO, Optional, Callable, Generic, TypeVar, Type, Union

from relic.core.lazyio import read_chunks

from relic.sga.core.errors import (
    HashMismatchError,
    Md5MismatchError,
    Crc32MismatchError,
    Sha1MismatchError,
)

_T = TypeVar("_T")

Hashable = Union[BinaryIO, bytes, bytes]


# TODO
#   At the time it felt like having an objec that stores the window informatino was a good thing
#   However; we never cache the windowed hashers because we rarely can garuntee that the window is the same size
#   Furthermore, we rarely want to hash multiple times and if the result is cached, we never re-call our hash object
#       I'd say keep the functionality, but make these 'classmethods'
class _Hasher(Generic[_T]):
    HASHER_NAME = "Hash"

    def __init__(
        self,
        hash_func: Callable[[Hashable], _T],
        error_cls: Type[HashMismatchError] = HashMismatchError,
    ):
        self._hasher = hash_func
        # Allows us to override the error class,
        # useful for catching different hash mismatches
        # and handling them differently
        self._error = error_cls

    def __call__(self, stream: Hashable):
        return self.hash(stream=stream)

    def hash(self, stream: Hashable) -> _T:
        return self._hasher(stream)

    def check(self, stream: Hashable, expected: _T):
        result = self.hash(stream=stream)
        return result == expected

    def validate(self, stream: Hashable, expected: _T, *, name: Optional[str] = None):
        result = self.hash(stream=stream)
        if result != expected:
            raise self._error(name or self.HASHER_NAME, result, expected)


class md5(_Hasher[bytes]):
    HASHER_NAME = "MD5"

    def __init__(
        self,
        start: Optional[int] = None,
        size: Optional[int] = None,
        eigen: Optional[bytes] = None,
    ):
        func = self.__factory(start=start, size=size, eigen=eigen)
        super().__init__(func, error_cls=Md5MismatchError)

    @staticmethod
    def __factory(
        start: Optional[int] = None,
        size: Optional[int] = None,
        eigen: Optional[bytes] = None,
    ) -> Callable[[Hashable], bytes]:
        def _md5(stream: Hashable) -> bytes:
            hasher = (
                hashlib.md5(eigen, usedforsecurity=False)
                if eigen is not None
                else hashlib.md5(usedforsecurity=False)
            )
            for chunk in read_chunks(stream, start, size):
                hasher.update(chunk)
            return hasher.digest()

        return _md5


class crc32(_Hasher[int]):
    HASHER_NAME = "CRC 32"

    def __init__(
        self,
        start: Optional[int] = None,
        size: Optional[int] = None,
        eigen: Optional[int] = None,
    ):
        func = self.__factory(start=start, size=size, eigen=eigen)
        super().__init__(func, error_cls=Crc32MismatchError)

    @staticmethod
    def __factory(
        start: Optional[int] = None,
        size: Optional[int] = None,
        eigen: Optional[int] = None,
    ) -> Callable[[Hashable], int]:
        def _crc32(stream: Hashable) -> int:
            crc = eigen if eigen is not None else 0
            for chunk in read_chunks(stream, start, size):
                crc = zlib.crc32(chunk, crc)
            return crc

        return _crc32


class sha1(_Hasher[bytes]):
    HASHER_NAME = "SHA-1"

    def __init__(
        self,
        start: Optional[int] = None,
        size: Optional[int] = None,
        eigen: Optional[bytes] = None,
    ):
        func = self.__factory(start=start, size=size, eigen=eigen)
        super().__init__(func, error_cls=Sha1MismatchError)

    @staticmethod
    def __factory(
        start: Optional[int] = None,
        size: Optional[int] = None,
        eigen: Optional[bytes] = None,
    ) -> Callable[[Hashable], bytes]:
        def _sha1(stream: Hashable) -> bytes:
            hasher = (
                hashlib.sha1(eigen, usedforsecurity=False)
                if eigen is not None
                else hashlib.sha1(usedforsecurity=False)
            )
            for chunk in read_chunks(stream, start, size):
                hasher.update(chunk)
            return hasher.digest()

        return _sha1
