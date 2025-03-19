from typing import Optional, List

import pytest

from relic.sga.core import Version
from relic.sga.core.errors import (
    MagicMismatchError,
    VersionMismatchError,
    VersionNotSupportedError,
    DecompressedSizeMismatch,
)


@pytest.mark.parametrize("received", [None, b"Good"])
@pytest.mark.parametrize("expected", [None, b"Bad"])
def test_magic_mismatch_error(received: Optional[bytes], expected: Optional[bytes]):
    # Ensure init does not raise error
    _ = MagicMismatchError(received, expected)


@pytest.mark.parametrize("received", [None, Version(-1)])
@pytest.mark.parametrize("expected", [None, Version(0)])
def test_version_mismatch_error(
    received: Optional[Version], expected: Optional[Version]
):
    # Ensure init does not raise error
    _ = VersionMismatchError(received, expected)


@pytest.mark.parametrize("received", [Version(-1)])
@pytest.mark.parametrize("allowed", [[], [Version(0)]])
class TestVersionNotSupportedError:
    def test_init(self, received: Version, allowed: List[Version]):
        _ = VersionNotSupportedError(received, allowed)

    def test_str(self, received: Version, allowed: List[Version]):
        err = VersionNotSupportedError(received, allowed)
        result = str(err)
        assert isinstance(result, str)


@pytest.mark.parametrize("received", [(None,), (2,)])
@pytest.mark.parametrize("expected", [(None,), (10,)])
def test_decompressed_size_mismatch(
    received: Optional[int], expected: Optional[int]
) -> None:
    _ = DecompressedSizeMismatch(received, expected), expected
