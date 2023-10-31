import hashlib
import zlib
from typing import BinaryIO, Optional, Generic, TypeVar, Type, Union, Protocol

from relic.core.lazyio import read_chunks

from relic.sga.core.errors import (
    HashMismatchError,
    Md5MismatchError,
    Crc32MismatchError,
    Sha1MismatchError,
)

_T_CON = TypeVar("_T_CON", contravariant=True)
_T = TypeVar("_T")

Hashable = Union[BinaryIO, bytes, bytearray]


class _HasherHashFunc(Protocol[_T]):
    def __call__(
        self,
        stream: Hashable,
        *,
        start: Optional[int] = None,
        size: Optional[int] = None,
        eigen: Optional[_T] = None,
    ) -> _T:
        raise NotImplementedError


class Hasher(Generic[_T]):
    def __init__(
        self,
        hasher_name: str,
        hash_func: _HasherHashFunc[_T],
        default_err_cls: Type[HashMismatchError[_T]] = HashMismatchError,
    ):
        self._hasher_name = hasher_name
        self._default_err_cls = default_err_cls
        self._hash_func = hash_func
        if not hasattr(self, "__name__"):
            self.__name__ = self._hasher_name

    def __call__(
        self,
        stream: Hashable,
        *,
        start: Optional[int] = None,
        size: Optional[int] = None,
        eigen: Optional[_T] = None,
    ) -> _T:
        return self.hash(stream=stream, start=start, size=size, eigen=eigen)

    def hash(
        self,
        stream: Hashable,
        *,
        start: Optional[int] = None,
        size: Optional[int] = None,
        eigen: Optional[_T] = None,
    ) -> _T:
        return self._hash_func(stream=stream, start=start, size=size, eigen=eigen)

    def check(
        self,
        stream: Hashable,
        expected: _T,
        *,
        start: Optional[int] = None,
        size: Optional[int] = None,
        eigen: Optional[_T] = None,
    ) -> bool:
        result = self.hash(stream=stream, start=start, size=size, eigen=eigen)
        return result == expected

    def validate(
        self,
        stream: Hashable,
        expected: _T,
        *,
        start: Optional[int] = None,
        size: Optional[int] = None,
        eigen: Optional[_T] = None,
        err_cls: Optional[Type[HashMismatchError[_T]]] = None,
        name: Optional[str] = None,
    ) -> None:
        result = self.hash(stream=stream, start=start, size=size, eigen=eigen)
        if result != expected:
            if err_cls is None:
                err_cls = self._default_err_cls

            raise err_cls(
                name if name is not None else self._hasher_name, result, expected
            )


def _md5(
    stream: Hashable,
    *,
    start: Optional[int] = None,
    size: Optional[int] = None,
    eigen: Optional[bytes] = None,
) -> bytes:
    hasher = (
        hashlib.md5(eigen, usedforsecurity=False)
        if eigen is not None
        else hashlib.md5(usedforsecurity=False)
    )
    for chunk in read_chunks(stream, start, size):
        hasher.update(chunk)
    return hasher.digest()


def _crc32(
    stream: Hashable,
    *,
    start: Optional[int] = None,
    size: Optional[int] = None,
    eigen: Optional[int] = None,
) -> int:
    crc = eigen if eigen is not None else 0
    for chunk in read_chunks(stream, start, size):
        crc = zlib.crc32(chunk, crc)
    return crc


def _sha1(
    stream: Hashable,
    *,
    start: Optional[int] = None,
    size: Optional[int] = None,
    eigen: Optional[bytes] = None,
) -> bytes:
    hasher = (
        hashlib.sha1(eigen, usedforsecurity=False)
        if eigen is not None
        else hashlib.sha1(usedforsecurity=False)
    )
    for chunk in read_chunks(stream, start, size):
        hasher.update(chunk)
    return hasher.digest()


# Create hashers bound to their hash method
md5 = Hasher("MD5", _md5, Md5MismatchError)
crc32 = Hasher("CRC-32", _crc32, Crc32MismatchError)
sha1 = Hasher("SHA-1", _sha1, Sha1MismatchError)

__all__ = ["Hashable", "md5", "crc32", "sha1", "Hasher"]
