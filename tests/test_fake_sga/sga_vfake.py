import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, List
from urllib.parse import ParseResult

from fs.osfs import OSFS
from relic.core import CLI

from relic.sga.core.essencefs import EssenceFS, EssenceFsOpener, EssenceFsOpenerPlugin

from relic.sga.core import Version, MAGIC_WORD

from src.relic.sga.core.essencefs.opener import registry
from src.relic.sga.core.serialization import SgaFile

FakeVersion = Version(0,0)


class FakeSgaFile(SgaFile):
    PATH_SIZE = (12,2)
    PATH_START = 14
    @property
    def realpath(self) -> str:
        size = self._serializer.uint16.read(*self.PATH_SIZE,byteorder="little", signed=False)
        value = self._serializer.c_string.read(self.PATH_START,size,encoding="ascii")
        return value
    @realpath.setter
    def realpath(self, value:str):
        size = len(value)
        self._serializer.uint16.write(size,*self.PATH_SIZE,byteorder="little", signed=False)
        self._serializer.c_string.write(value,self.PATH_START,size,encoding="ascii")

    @classmethod
    def make_binary_handle(cls, path:str) -> BytesIO:
        buffer = b"\0" * (cls.PATH_START + len(path))
        handle = None
        try:
            handle = BytesIO(buffer)
            sga = FakeSgaFile(handle) # Can't write the magic... I understand why i made it read only but also UGHHH
            MAGIC_WORD.write(handle)
            sga.version = FakeVersion
            sga.realpath = path
            handle.seek(0)
            return handle
        except:
            if handle is not None:
                handle.close()
            raise

class FakeSga(EssenceFS):
    def __init__(self, osfs: OSFS):
        super().__init__()
        self._backing = osfs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._backing.close()

    def getinfo(self, path, namespaces=None):
        return self._backing.getinfo(path,namespaces)

    def listdir(self, path):
        return self._backing.listdir(path)

    def makedir(self, path, permissions=None, recreate=False):
        return self._backing.makedir(path,permissions,recreate)

    def openbin(self, path, mode="r", buffering=-1, **options):
        return self._backing.openbin(path,mode,buffering,**options)

    def remove(self, path):
        return self._backing.remove(path)

    def removedir(self, path):
        return self._backing.removedir(path)

    def setinfo(self, path, info):
        return self._backing.setinfo(path,info)
class FakeFileOpener(EssenceFsOpenerPlugin):
    @property
    def protocols(self) -> List[str]:
        return ["sga-fake"]

    @property
    def versions(self) -> List[Version]:
        return [FakeVersion]

    def __repr__(self) -> str:
        return "Fake Opener"

    def open_fs(
            self,
            fs_url: str,
            parse_result: ParseResult,
            writeable: bool,
            create: bool,
            cwd: str,
    ) -> FakeSga:
        with open(parse_result.path, "rb") as h:
            fake = FakeSgaFile(h)
            path = fake.realpath
            return FakeSga(OSFS(path))


ROOT_TEST_FOLDER = Path(__file__).parent.parent


def test_cli_tree():
    registry.register(FakeVersion, FakeFileOpener())
    fake_file_name = None
    try:

        with tempfile.NamedTemporaryFile("wb", delete=False) as fake_file:
            fake_file_name = fake_file.name
            with FakeSgaFile.make_binary_handle(str(ROOT_TEST_FOLDER)) as data_handle:
                fake_file.write(data_handle.read())

        result = CLI.run_with("sga","tree",fake_file_name)
        assert result == 0

    finally:
        try:
            os.unlink(fake_file_name)
        except:
            pass

