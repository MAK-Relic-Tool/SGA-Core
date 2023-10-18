import unittest

import fs
from fs.test import FSTestCases

from relic.sga.core.filesystem import EssenceFS, _EssenceDriveFS


class TestEssenceFS(FSTestCases, unittest.TestCase):
    def make_fs(self):
        essence_fs = EssenceFS()
        # EssenceFS shouldn't be writeable by default;
        #   being an emulator for Window's hard drives.
        #       With no 'drive' installed, there's nothing to write to!
        essence_fs.add_fs("data", _EssenceDriveFS("data", "test"), True)
        return essence_fs


class TestEssenceDriveFS(FSTestCases, unittest.TestCase):
    def make_fs(self):
        return _EssenceDriveFS("data", "test")


class TestOpener:
    def test_open_fs(self):
        with fs.open_fs("sga://", create=True) as sga:
            pass
