import unittest

from fs.test import FSTestCases

from relic.sga.core.filesystem import EssenceFS, _EssenceDriveFS


class TestEssenceFS(FSTestCases, unittest.TestCase):
    def make_fs(self):
        essence_fs = EssenceFS()
        essence_fs.add_fs("data", _EssenceDriveFS(), True)
        return essence_fs


class TestEssenceDriveFS(FSTestCases, unittest.TestCase):
    def make_fs(self):
        return _EssenceDriveFS()
