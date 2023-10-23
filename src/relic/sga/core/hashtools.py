import hashlib
import zlib
from typing import BinaryIO, Optional, Callable, Generic, TypeVar, Type

from relic.core.errors import MismatchError

from relic.sga.core.lazyio import read_chunks

T = TypeVar("T")


class HashMismatchError(MismatchError[T], Generic[T]):
    """
    A sentinel class for catching all hash mismatch errors.
    """

    ...


class _Hasher(Generic[T]):
    HASHER_NAME = "Hash"

    def __init__(
        self,
        hash_func: Callable[[BinaryIO], T],
        error_cls: Type[HashMismatchError] = HashMismatchError,
    ):
        self._hasher = hash_func
        # Allows us to override the error class,
        # useful for catching different hash mismatches
        # and handling them differently
        self._error = error_cls

    def __call__(self, stream: BinaryIO):
        return self.hash(stream=stream)

    def hash(self, stream: BinaryIO) -> T:
        return self._hasher(stream)

    def check(self, stream: BinaryIO, expected: T):
        result = self.hash(stream=stream)
        return result == expected

    def validate(self, stream: BinaryIO, expected: T, *, name: Optional[str] = None):
        result = self.hash(stream=stream)
        if result != expected:
            raise self._error(name or self.HASHER_NAME, result, expected)


class Md5MismatchError(HashMismatchError[bytes]):  #
    ...


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
    ) -> Callable[[BinaryIO], bytes]:
        def _md5(stream: BinaryIO) -> bytes:
            hasher = (
                hashlib.md5(eigen, usedforsecurity=False)
                if eigen is not None
                else hashlib.md5(usedforsecurity=False)
            )
            for chunk in read_chunks(stream, start, size):
                hasher.update(chunk)
            return hasher.digest()

        return _md5


class Crc32MismatchError(HashMismatchError[int]):
    ...


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
    ) -> Callable[[BinaryIO], int]:
        def _crc32(stream: BinaryIO) -> int:
            crc = eigen if eigen is not None else 0
            for chunk in read_chunks(stream, start, size):
                crc = zlib.crc32(chunk, crc)
            return crc

        return _crc32


class Sha1MismatchError(HashMismatchError[bytes]):  #
    ...


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
    ) -> Callable[[BinaryIO], bytes]:
        def _sha1(stream: BinaryIO) -> bytes:
            hasher = (
                hashlib.sha1(eigen, usedforsecurity=False)
                if eigen is not None
                else hashlib.sha1(usedforsecurity=False)
            )
            for chunk in read_chunks(stream, start, size):
                hasher.update(chunk)
            return hasher.digest()

        return _sha1
