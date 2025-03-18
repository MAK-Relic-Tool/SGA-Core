import fs.opener.errors
import pytest
from relic.core.errors import RelicToolError

from tests.dummy_essencefs import register_randomfs_opener, RandomEssenceFsOpener
from relic.sga.core import MAGIC_WORD, Version
from relic.sga.core.errors import VersionNotSupportedError
from relic.sga.core.essencefs.opener import registry
from relic.sga.core.serialization import VersionSerializer
from tests.util import TempFileHandle


class TestEssenceFsOpener:
    def test_open_fs_no_mem_create(self):
        try:
            fs.open_fs("", create=True, default_protocol="sga")
        except RelicToolError:
            pass
        else:
            pytest.fail('Opening FS with "" & create=True should fail')

    def test_open_fs_no_empty_fsurl(self):
        try:
            fs.open_fs("", create=False, default_protocol="sga")
        except fs.opener.errors.OpenerError:
            pass
        else:
            pytest.fail('Opening FS with "" & create=False should fail')

    def test_open_version_not_supported(self):
        register_randomfs_opener()
        with TempFileHandle() as h:
            with h.open("wb") as w:
                MAGIC_WORD.write(w)
                VersionSerializer.write(w, Version(920, 2004))
            try:
                with fs.open_fs(h.path, default_protocol="sga") as _:
                    pass
            except VersionNotSupportedError:
                pass
            else:
                pytest.fail('Opening FS with "" & create=False should fail')

    def test_opener_instance_compatible(self):
        # Tests that the opener will work properly if the opener is an instance
        registry.register(Version(920, 2004), RandomEssenceFsOpener())
        with TempFileHandle() as h:
            with h.open("wb") as w:
                MAGIC_WORD.write(w)
                VersionSerializer.write(w, Version(920, 2004))
                w.write(b"beef")  # 4-byte seed; required for randomfs
            with fs.open_fs(h.path, default_protocol="sga") as _:
                pass
