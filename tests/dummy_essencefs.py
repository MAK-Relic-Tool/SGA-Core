import random
from typing import Iterator, Tuple, List, BinaryIO, Any, Dict

import fs.opener.registry
from fs import open_fs
from fs.base import FS
from fs.memoryfs import MemoryFS
from fs.opener.parse import ParseResult

from relic.sga.core import Version, MAGIC_WORD
from relic.sga.core.essencefs import EssenceFS
from relic.sga.core.essencefs.opener import registry, EssenceFsOpenerPlugin
from relic.sga.core.serialization import VersionSerializer


class RandomEssenceFS(EssenceFS, MemoryFS):

    def iterate_fs(self) -> Iterator[Tuple[str, FS]]:
        yield ("dummy", self)

    def info_tree(self, **options: Any) -> Dict[str, Any]:
        return {
            "dummy": "I want to deprecate this, what was it supposed to accomplish?"
        }

    @classmethod
    def generate_random_fs(cls, seed: int):
        random.seed = seed
        rand_fs = cls()
        max_depth = random.randint(2, 8)  # Max depth prevent deep random file systems
        max_items = random.randint(4, 32)  # Max items prevent big random file systems
        items = 0

        def random_name():
            return random.randbytes(8).hex().lower()

        def random_file(parent: FS):
            nonlocal items
            if items > max_items:
                return
            items += 1
            name = random_name()
            with parent.openbin(name, "w") as h:
                size = random.randint(16, 8192)
                h.write(random.randbytes(size))

        def rand_folder(parent: FS, depth: int = 0):
            nonlocal items
            if depth > max_depth:
                return
            if items > max_items:
                return
            items += 1

            name = random_name()
            with parent.makedir(name) as subdir:
                items = random.randint(0, 4)
                files = random.randint(0, items)
                folders = items - files

                for _ in range(files):
                    random_file(subdir)
                for _ in range(folders):
                    rand_folder(subdir, depth + 1)

        rand_folder(rand_fs)
        return rand_fs


class RandomEssenceFsOpener(EssenceFsOpenerPlugin):

    @property
    def protocols(self) -> List[str]:
        return ["sga-random"]

    def open_fs(
        self,
        fs_url: str,
        parse_result: ParseResult,
        writeable: bool,
        create: bool,
        cwd: str,
    ) -> EssenceFS:
        with open(parse_result.resource, "rb") as h:
            MAGIC_WORD.validate(h)
            version = VersionSerializer.read(h)
            seed = int.from_bytes(h.read(4), "little", signed=False)
            return RandomEssenceFS.generate_random_fs(seed)

    @classmethod
    def write_file(cls, handle: BinaryIO, seed: int, version: Version):
        MAGIC_WORD.write(handle)
        VersionSerializer.write(handle, version)
        buffer = seed.to_bytes(4, "little", signed=False)
        handle.write(buffer)

    @property
    def versions(self) -> List[Version]:
        return []


RAND_SGAFS_VER = Version(0, 0)


def open_random_essencefs(seed: int):
    return open_fs(str(seed), default_protocol=RandomEssenceFsOpener.protocols[0])


def write_random_essencefs(handle: BinaryIO, seed: int):
    return RandomEssenceFsOpener.write_file(handle, seed, RAND_SGAFS_VER)


def register_randomfs_opener():
    registry.register(RAND_SGAFS_VER, RandomEssenceFsOpener)
