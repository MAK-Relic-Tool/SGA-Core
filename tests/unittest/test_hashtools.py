import dataclasses
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Type, Iterable, Tuple, Union
from hashlib import md5 as calc_md5, sha1 as calc_sha1
from zlib import crc32 as calc_crc32
import pytest

from relic.sga.core.errors import HashMismatchError
from relic.sga.core.hashtools import Hashable, _T, Hasher, md5, sha1, crc32


@dataclass
class HashArgs:
    stream: Hashable
    start: Optional[int] = None
    size: Optional[int] = None
    eigen: Optional[_T] = None

    @property
    def hash_kwargs(self):
        return dataclasses.asdict(self)


class ValidateArgs(HashArgs):
    err_cls: Optional[Type[HashMismatchError]] = None
    name: Optional[str] = None

    @property
    def hash_kwargs(self):
        hash_kwargs = super().hash_kwargs
        del hash_kwargs["err_cls"]
        del hash_kwargs["name"]
        return hash_kwargs

    @property
    def validate_kwargs(self):
        return super().hash_kwargs


_input_args = [
    (HashArgs(b"Alec Baldwin")),
    (HashArgs(b"Ben Afleck", start=4)),
    (HashArgs(b"Chris Pratt", size=5)),
    (HashArgs(b"Donald Glover", start=7, size=5)),
    (HashArgs(b"Eric Andre", eigen=5202012)),
    (HashArgs(b"Fraser, Branden", start=8, eigen=5041999)),
    (HashArgs(b"Gene Simmons", size=4, eigen=2181974)),
    (HashArgs(b"Hulk Hogan", start=5, size=3, eigen=1012012))
]


def _invert_hash(t: Union[bytes, int]):
    if isinstance(t, int):
        return t ^ 0xFFFFFFFF
    return bytes(v ^ 255 for v in t)


def _md5_from_input(items:Iterable[HashArgs]):
    TF = [True,False]
    for args in items:
        buffer:bytes = args.stream
        if args.start is not None:
            buffer =buffer[args.start:]
        if args.size is not None:
            buffer = buffer[:args.size]

        if args.eigen is not None:
            eigen = args.eigen.to_bytes(4,"little",signed=False)
        else:
            eigen = None

        hasher = calc_md5(usedforsecurity=False)
        if eigen is not None:
            hasher.update(eigen)
        hasher.update(buffer)
        calced = hasher.digest()
        for use_bytes in TF:
            for should_pass in TF:
                expected = _invert_hash(calced) if not should_pass else calced

                hash_args = HashArgs(BytesIO(args.stream) if not use_bytes else args.stream, args.start, args.size, eigen)

                yield md5, hash_args, expected, should_pass


def _sha1_from_input(items:Iterable[HashArgs]):
    TF = [True,False]
    for args in items:
        buffer:bytes = args.stream
        if args.start is not None:
            buffer =buffer[args.start:]
        if args.size is not None:
            buffer = buffer[:args.size]

        if args.eigen is not None:
            eigen = args.eigen.to_bytes(4,"little",signed=False) # Arbitary mapping of int to bytes; so long as the same eigen is used; the same results should be gotten
        else:
            eigen = None

        hasher = calc_sha1(usedforsecurity=False)
        if eigen is not None:
            hasher.update(eigen)
        hasher.update(buffer)
        calced = hasher.digest()
        for use_bytes in TF:
            for should_pass in TF:
                expected = _invert_hash(calced) if not should_pass else calced

                hash_args = HashArgs(BytesIO(args.stream) if not use_bytes else args.stream, args.start, args.size, eigen)

                yield sha1, hash_args, expected, should_pass


def _crc32_from_input(items:Iterable[HashArgs]):
    TF = [True,False]
    for args in items:
        buffer:bytes = args.stream
        if args.start is not None:
            buffer =buffer[args.start:]
        if args.size is not None:
            buffer = buffer[:args.size]

        eigen = args.eigen

        if eigen is None:
            calced = calc_crc32(buffer)
        else:
            calced = calc_crc32(buffer,eigen)

        for use_bytes in TF:
            for should_pass in TF:
                expected = _invert_hash(calced) if not should_pass else calced

                hash_args = HashArgs(BytesIO(args.stream) if not use_bytes else args.stream, args.start, args.size, eigen)

                yield crc32, hash_args, expected, should_pass


def _convert_input_2_tests(items: Iterable[HashArgs]):
    items = list(items)
    yield from _md5_from_input(items)
    yield from _sha1_from_input(items)
    yield from _crc32_from_input(items)




_VALID_HASHER_TESTS = list(_convert_input_2_tests(_input_args))
_VALID_HASHER_IDS = list(f"{_[0].__name__} ~ {_[1]} ~ {_[2]} ~ {'Match' if _[3] else 'MisMatch'}" for _ in _VALID_HASHER_TESTS)
_HASHER_TESTS = list((a,b,c) for (a,b,c,d) in _VALID_HASHER_TESTS if d) # list((_[0], _[1], _[2]) for _ in _VALID_HASHER_TESTS if _[3] is False)
_HASHER_IDS = list(f"{_[0].__name__} ~ {_[1]} ~ {_[2]}" for _ in _HASHER_TESTS)


@pytest.mark.parametrize(["hasher", "args", "expected"], _HASHER_TESTS, ids = _HASHER_IDS)
def test_hasher_hash(hasher: Hasher, args: HashArgs, expected: _T):
    if hasattr(args.stream,"seek"):
        args.stream.seek(0)
    result = hasher.hash(**args.hash_kwargs)
    assert result == expected


@pytest.mark.parametrize(["hasher", "args", "expected", "expected_result"], _VALID_HASHER_TESTS, ids=_VALID_HASHER_IDS)
def test_hasher_check(hasher: Hasher, args: HashArgs, expected: _T, expected_result: bool):
    if hasattr(args.stream, "seek"):
        args.stream.seek(0)

    result = hasher.check(**args.hash_kwargs, expected=expected)
    assert result == expected_result


@pytest.mark.parametrize(["hasher", "args", "expected", "expected_result"], _VALID_HASHER_TESTS, ids=_VALID_HASHER_IDS)
def test_hasher_validate(hasher: Hasher, args: ValidateArgs, expected: _T, expected_result: bool):
    if hasattr(args.stream, "seek"):
        args.stream.seek(0)

    try:
        hasher.validate(**args.hash_kwargs, expected=expected)
    except HashMismatchError:
        assert expected_result is False
    else:
        assert expected_result is True


def test_hasher_validate_err_cls(hasher: Hasher, args: ValidateArgs, expected_failure: _T,
                                 expected_err_cls: Type[HashMismatchError]):
    if hasattr(args.stream,"seek"):
        args.stream.seek(0)

    try:
        hasher.validate(*args.hash_kwargs, expected=expected_failure)
    except expected_err_cls:
        pass
    except Exception as e:
        assert isinstance(e, expected_err_cls)
    else:
        pytest.fail("Hasher did not raise an error!")


def test_hasher_validate_err_name(hasher: Hasher, args: ValidateArgs, expected_failure: _T, expected_name: str):
    if hasattr(args.stream,"seek"):
        args.stream.seek(0)
    try:
        hasher.validate(*args.validate_kwargs, expected=expected_failure)
    except HashMismatchError as e:
        assert e.name is expected_name
    else:
        pytest.fail("Hasher did not raise a HashMismatchError!")
