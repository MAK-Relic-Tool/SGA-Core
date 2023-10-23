import itertools

import pytest

from relic.sga.core.definitions import Version

# Chunky versions start at 1
# Max i've seen is probably 16ish?
# Minor has a lot mor variety than Chunky; so we test a bit more
_VERSION_MAJORS = range(1, 11)  # So far we only go up to V10
_VERSION_MINORS = [0, 1]  # Allegedly CoHO was v4.1 so... we do 0,1


_VERSION_ARGS = list(itertools.product(_VERSION_MAJORS, _VERSION_MINORS))
_VERSIONS = [Version(*a) for a in _VERSION_ARGS]
_VERSION_IDS = [f"V{v.major}.{v.minor}" for v in _VERSIONS]


class TestVersion:
    @pytest.fixture(params=_VERSIONS, ids=_VERSION_IDS)
    def this(self, request):
        return request.param

    @pytest.fixture(params=_VERSIONS, ids=_VERSION_IDS)
    def other(self, request):
        return request.param

    @staticmethod
    def lt(self: Version, other: Version) -> bool:
        if self.major > other.major:
            return False
        if self.major == other.major:
            return self.minor < other.minor
        return self.major < other.major

    @classmethod
    def lteq(cls, self: Version, other: Version) -> bool:
        return not cls.gt(self, other)

    @staticmethod
    def gt(self: Version, other: Version) -> bool:
        if self.major < other.major:
            return False
        if self.major == other.major:
            return self.minor > other.minor
        return self.major > other.major

    @classmethod
    def gteq(cls, self: Version, other: Version) -> bool:
        return not cls.lt(self, other)

    @staticmethod
    def eq(self: Version, other: Version) -> bool:
        return self.major == other.major and self.minor == other.minor

    @classmethod
    def neq(cls, self: Version, other: Version) -> bool:
        return not cls.eq(self, other)

    def test_lt(self, this, other):
        result = this < other
        expected = TestVersion.lt(this, other)
        assert result == expected

    def test_lteq(self, this, other):
        result = this <= other
        expected = TestVersion.lteq(this, other)
        assert result == expected

    def test_gt(self, this, other):
        result = this > other
        expected = TestVersion.gt(this, other)
        assert result == expected

    def test_gteq(self, this, other):
        result = this >= other
        expected = TestVersion.gteq(this, other)
        assert result == expected

    def test_eq(self, this, other):
        result = this == other
        expected = TestVersion.eq(this, other)
        assert result == expected

    def test_neq(self, this, other):
        result = this != other
        expected = TestVersion.neq(this, other)
        assert result == expected
