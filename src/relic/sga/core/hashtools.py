import hashlib
import zlib
from typing import BinaryIO, Optional, Callable, Any, Union, Generic, TypeVar

from relic.sga.core.lazyio import read_chunks

T = TypeVar("T")


class _Hasher(Generic[T]):
    def __init__(self, hash_func: Callable[[BinaryIO], T]):
        self._hasher = hash_func

    def __call__(self, stream: BinaryIO):
        return self.hash(stream=stream)

    def hash(self, stream: BinaryIO) -> T:
        return self._hasher(stream)

    def validate(self, stream: BinaryIO, expected: T):
        result = self.hash(stream=stream)
        return result == expected


class md5(_Hasher[bytes]):
    def __init__(self, start: Optional[int] = None, size: Optional[int] = None, eigen: Optional[bytes] = None):
        func = self.__factory(start=start, size=size, eigen=eigen)
        super().__init__(func)

    @staticmethod
    def __factory(start: Optional[int] = None, size: Optional[int] = None,
                  eigen: Optional[bytes] = None) -> Callable[[BinaryIO], bytes]:
        def _md5(stream: BinaryIO) -> bytes:
            hasher = hashlib.md5(eigen, usedforsecurity=False) if eigen is not None else hashlib.md5(
                usedforsecurity=False)
            for chunk in read_chunks(stream, start, size):
                hasher.update(chunk)
            return hasher.digest()

        return _md5


class crc32(_Hasher[int]):
    def __init__(self, start: Optional[int] = None, size: Optional[int] = None, eigen: Optional[int] = None):
        func = self.__factory(start=start, size=size, eigen=eigen)
        super().__init__(func)

    @staticmethod
    def __factory(start: Optional[int] = None, size: Optional[int] = None,
                  eigen: Optional[int] = None) -> Callable[[BinaryIO], int]:
        def _crc32(stream: BinaryIO) -> int:
            crc = eigen if eigen is not None else 0
            for chunk in read_chunks(stream, start, size):
                crc = zlib.crc32(chunk, crc)
            return crc

        return _crc32


class sha1(_Hasher[bytes]):
    def __init__(self, start: Optional[int] = None, size: Optional[int] = None, eigen: Optional[bytes] = None):
        func = self.__factory(start=start, size=size, eigen=eigen)
        super().__init__(func)

    @staticmethod
    def __factory(start: Optional[int] = None, size: Optional[int] = None,
                  eigen: Optional[bytes] = None) -> Callable[[BinaryIO], bytes]:
        def _sha1(stream: BinaryIO) -> bytes:
            hasher = hashlib.sha1(eigen, usedforsecurity=False) if eigen is not None else hashlib.sha1(
                usedforsecurity=False)
            for chunk in read_chunks(stream, start, size):
                hasher.update(chunk)
            return hasher.digest()

        return _sha1
