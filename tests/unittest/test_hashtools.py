import dataclasses
from dataclasses import dataclass
from hashlib import md5 as calc_md5, sha1 as calc_sha1
from io import BytesIO
from typing import Optional, Type, Iterable, Union, Callable
from zlib import crc32 as calc_crc32

import pytest

from relic.sga.core.errors import (
    HashMismatchError,
    Crc32MismatchError,
    Sha1MismatchError,
    Md5MismatchError,
)
from relic.sga.core.hashtools import Hashable, _T, Hasher, md5, sha1, crc32


@dataclass
class HashArgs:
    stream: Hashable
    start: Optional[int] = None
    size: Optional[int] = None
    eigen: Optional[_T] = None
    err_cls: Optional[Type[HashMismatchError]] = None
    name: Optional[str] = None

    @property
    def hash_kwargs(self):
        hash_kwargs = dataclasses.asdict(self)
        del hash_kwargs["err_cls"]
        del hash_kwargs["name"]
        return hash_kwargs

    @property
    def validate_kwargs(self):
        return dataclasses.asdict(self)


def _invert_hash(t: Union[bytes, int]):
    if isinstance(t, int):
        return t ^ 0xFFFFFFFF
    return bytes(v ^ 255 for v in t)


def _hasher_from_input(
    items: Iterable[HashArgs],
    parse_eigen: Callable[[Optional[int]], Optional[Union[int, bytes]]],
    calc_hash: Callable[[bytes, Union[int, bytes]], Union[bytes, int]],
    hasher: Hasher,
    err_cls: Type[HashMismatchError],
):
    TF = [True, False]
    for args in items:
        buffer: bytes = args.stream
        if args.start is not None:
            buffer = buffer[args.start :]
        if args.size is not None:
            buffer = buffer[: args.size]

        eigen = parse_eigen(args.eigen)
        calced = calc_hash(buffer, eigen)
        for use_bytes in TF:
            for should_pass in TF:
                expected = _invert_hash(calced) if not should_pass else calced

                hash_args = HashArgs(
                    BytesIO(args.stream) if not use_bytes else args.stream,
                    args.start,
                    args.size,
                    eigen,
                )

                yield hasher, hash_args, expected, should_pass, err_cls


def _md5_from_input(items: Iterable[HashArgs]):
    def _eigen(e: Optional[int]) -> Optional[bytes]:
        if e is not None:
            return e.to_bytes(4, "little", signed=False)
        else:
            return None

    def _hash(b: bytes, e: Optional[bytes]) -> bytes:
        hasher = calc_md5(usedforsecurity=False)
        if e is not None:
            hasher.update(e)
        hasher.update(b)
        return hasher.digest()

    yield from _hasher_from_input(items, _eigen, _hash, md5, Md5MismatchError)


def _sha1_from_input(items: Iterable[HashArgs]):
    def _eigen(e: Optional[int]) -> Optional[bytes]:
        if e is not None:
            return e.to_bytes(4, "little", signed=False)
        else:
            return None

    def _hash(b: bytes, e: Optional[bytes]) -> bytes:
        hasher = calc_sha1(usedforsecurity=False)
        if e is not None:
            hasher.update(e)
        hasher.update(b)
        return hasher.digest()

    yield from _hasher_from_input(items, _eigen, _hash, sha1, Sha1MismatchError)


def _crc32_from_input(items: Iterable[HashArgs]):
    def _eigen(e: Optional[int]) -> Optional[int]:
        return e

    def _hash(b: bytes, e: Optional[int]) -> int:
        if e is None:
            return calc_crc32(b)
        else:
            return calc_crc32(b, e)

    yield from _hasher_from_input(items, _eigen, _hash, crc32, Crc32MismatchError)


def _convert_input_2_tests(items: Iterable[HashArgs]):
    items = list(items)
    yield from _md5_from_input(items)
    yield from _sha1_from_input(items)
    yield from _crc32_from_input(items)


_input_args = [
    (HashArgs(b"Alec Baldwin")),
    (HashArgs(b"Ben Afleck", start=4)),
    (HashArgs(b"Chris Pratt", size=5)),
    (HashArgs(b"Donald Glover", start=7, size=5)),
    (HashArgs(b"Eric Andre", eigen=5202012)),
    (HashArgs(b"Fraser, Branden", start=8, eigen=5041999)),
    (HashArgs(b"Gene Simmons", size=4, eigen=2181974)),
    (HashArgs(b"Hulk Hogan", start=5, size=3, eigen=1012012)),
]

_TEST_DATA = list(_convert_input_2_tests(_input_args))

_HASHER_TESTS = list(
    (hasher, args, buffer)
    for (hasher, args, buffer, passing, _) in _TEST_DATA
    if passing
)
_HASHER_TEST_IDS = list(f"{_[0].__name__} ~ {_[1]} ~ {_[2]}" for _ in _HASHER_TESTS)

_HASHER_CHECK_TESTS = list(
    (hasher, args, buffer, passing) for (hasher, args, buffer, passing, _) in _TEST_DATA
)
_HASHER_CHECK_TEST_IDS = list(
    f"{_[0].__name__} ~ {_[1]} ~ {_[2]} ~ {'Match' if _[3] else 'MisMatch'}"
    for _ in _HASHER_CHECK_TESTS
)

_HASHER_VALIDATE_ERR_TESTS = list(
    (hasher, args, buffer, err_cls)
    for (hasher, args, buffer, passing, err_cls) in _TEST_DATA
    if not passing
)
_HASHER_VALIDATE_ERR_IDS = list(
    f"{_[0].__name__} ~ {_[1]} ~ {_[2]} ~ {_[3]}" for _ in _HASHER_VALIDATE_ERR_TESTS
)

_HASHER_VALIDATE_ERR_NAME_TESTS = list(
    (hasher, args, buffer, hasher._hasher_name)
    for (hasher, args, buffer, passing, _) in _TEST_DATA
    if not passing
)
_HASHER_VALIDATE_ERR_NAME_IDS = list(
    f"{_[0].__name__} ~ {_[1]} ~ {_[2]} ~ {_[3]}"
    for _ in _HASHER_VALIDATE_ERR_NAME_TESTS
)


@pytest.mark.parametrize(
    ["hasher", "args", "expected"], _HASHER_TESTS, ids=_HASHER_TEST_IDS
)
def test_hasher_hash(hasher: Hasher, args: HashArgs, expected: _T):
    if hasattr(args.stream, "seek"):
        args.stream.seek(0)
    result = hasher.hash(**args.hash_kwargs)
    assert result == expected


@pytest.mark.parametrize(
    ["hasher", "args", "expected", "expected_result"],
    _HASHER_CHECK_TESTS,
    ids=_HASHER_CHECK_TEST_IDS,
)
def test_hasher_check(
    hasher: Hasher, args: HashArgs, expected: _T, expected_result: bool
):
    if hasattr(args.stream, "seek"):
        args.stream.seek(0)

    result = hasher.check(**args.hash_kwargs, expected=expected)
    assert result == expected_result


@pytest.mark.parametrize(
    ["hasher", "args", "expected", "expected_result"],
    _HASHER_CHECK_TESTS,
    ids=_HASHER_CHECK_TEST_IDS,
)
def test_hasher_validate(
    hasher: Hasher, args: HashArgs, expected: _T, expected_result: bool
):
    if hasattr(args.stream, "seek"):
        args.stream.seek(0)

    try:
        hasher.validate(**args.hash_kwargs, expected=expected)
    except HashMismatchError:
        assert expected_result is False
    else:
        assert expected_result is True


@pytest.mark.parametrize(
    ["hasher", "args", "expected_failure", "expected_err_cls"],
    _HASHER_VALIDATE_ERR_TESTS,
    ids=_HASHER_VALIDATE_ERR_IDS,
)
def test_hasher_validate_err_cls(
    hasher: Hasher,
    args: HashArgs,
    expected_failure: _T,
    expected_err_cls: Type[HashMismatchError],
):
    if hasattr(args.stream, "seek"):
        args.stream.seek(0)

    try:
        hasher.validate(**args.hash_kwargs, expected=expected_failure)
    except expected_err_cls:
        pass
    except Exception as e:
        assert isinstance(e, expected_err_cls)
    else:
        pytest.fail("Hasher did not raise an error!")


@pytest.mark.parametrize(
    ["hasher", "args", "expected_failure", "expected_name"],
    _HASHER_VALIDATE_ERR_NAME_TESTS,
    ids=_HASHER_VALIDATE_ERR_NAME_IDS,
)
def test_hasher_validate_err_name(
    hasher: Hasher, args: HashArgs, expected_failure: _T, expected_name: str
):
    if hasattr(args.stream, "seek"):
        args.stream.seek(0)
    try:
        hasher.validate(**args.validate_kwargs, expected=expected_failure)
    except HashMismatchError as e:
        assert e.name is expected_name
    else:
        pytest.fail("Hasher did not raise a HashMismatchError!")
