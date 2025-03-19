from __future__ import annotations

from io import BytesIO, StringIO
from typing import (
    Optional,
    TypeVar,
    BinaryIO,
    Tuple,
    List,
    Type,
)

import pytest
from relic.core.errors import RelicToolError

import relic.sga.core.serialization as module
from relic.sga.core.definitions import Version, MAGIC_WORD

_T = TypeVar("_T")

_NULL_PTR = (None, None)


@pytest.mark.parametrize(
    ["parent", "default", "expected"],
    [
        ("Parent", "Default", "Parent"),
        (None, "Default", "Default"),
    ],
)
def test_safe_get_parent_name(
    parent: Optional[str], default: Optional[str], expected: Optional[str]
):
    with BytesIO() as stream:
        if parent is not None:
            stream.name = parent

    result = module._safe_get_parent_name(stream, default)
    assert result == expected


class FakeSgaTocHeader(module.SgaTocHeader):
    _DRIVE_POS = (0, 1)
    _DRIVE_COUNT = (1, 1)
    _FOLDER_POS = (2, 1)
    _FOLDER_COUNT = (3, 1)
    _FILE_POS = (4, 1)
    _FILE_COUNT = (5, 1)
    _NAME_POS = (6, 1)
    _NAME_COUNT = (7, 1)

    @classmethod
    def write_fake(
        cls,
        handle: BinaryIO,
        drive: Tuple[int, int],
        folder: Tuple[int, int],
        file: Tuple[int, int],
        name: Tuple[int, int],
    ) -> None:
        writer = FakeSgaTocHeader(handle)
        writer.drive.info = drive
        writer.folder.info = folder
        writer.file.info = file
        writer.name.info = name


@pytest.mark.parametrize("drive", [(0, 0), (0, 1)])
@pytest.mark.parametrize("folder", [(1, 2), (2, 3)])
@pytest.mark.parametrize("file", [(3, 4), (4, 5)])
@pytest.mark.parametrize("name", [(5, 6), (6, 7)])
def test_fake_sga_header(
    drive: Tuple[int, int],
    folder: Tuple[int, int],
    file: Tuple[int, int],
    name: Tuple[int, int],
):
    with BytesIO() as handle:
        FakeSgaTocHeader.write_fake(handle, drive, folder, file, name)
        handle.seek(0)
        reader = FakeSgaTocHeader(handle)
        assert reader.drive.info == drive
        assert reader.folder.info == folder
        assert reader.file.info == file
        assert reader.name.info == name


class FakeSgaTocDrive(module.SgaTocDrive):
    _ALIAS = (0, 4)
    _NAME = (4, 4)
    _FIRST_FOLDER = (9, 1)
    _LAST_FOLDER = (10, 1)
    _FIRST_FILE = (11, 1)
    _LAST_FILE = (12, 1)
    _ROOT_FOLDER = (13, 1)
    _SIZE = 14

    @classmethod
    def write_fake(
        cls,
        handle: BinaryIO,
        alias: str,
        name: str,
        folder: Tuple[int, int],
        file: Tuple[int, int],
        root_folder: int,
    ):
        writer = FakeSgaTocDrive(handle)
        writer.name = name
        writer.alias = alias
        writer.first_folder = folder[0]
        writer.last_folder = folder[1]
        writer.first_file = file[0]
        writer.last_file = file[1]
        writer.root_folder = root_folder


@pytest.mark.parametrize("alias", ["fake", "test"])
@pytest.mark.parametrize("name", ["dead", "beef"])
@pytest.mark.parametrize("folder", [(1, 2), (2, 3)])
@pytest.mark.parametrize("file", [(4, 0), (0, 4)])
@pytest.mark.parametrize("root_folder", [1, 2])
def test_fake_sga_toc_drive(
    alias: str,
    name: str,
    folder: Tuple[int, int],
    file: Tuple[int, int],
    root_folder: int,
):
    with BytesIO() as handle:
        FakeSgaTocDrive.write_fake(handle, alias, name, folder, file, root_folder)
        handle.seek(0)
        reader = FakeSgaTocDrive(handle)
        assert reader.name == name
        assert reader.alias == alias
        assert reader.first_folder == folder[0]
        assert reader.last_folder == folder[1]
        assert reader.first_file == file[0]
        assert reader.last_file == file[1]
        assert reader.root_folder == root_folder


class FakeSgaTocFolder(module.SgaTocFolder):
    _NAME_OFFSET = (0, 1)
    _SUB_FOLDER_START = (1, 1)
    _SUB_FOLDER_STOP = (2, 1)
    _FIRST_FILE = (3, 1)
    _LAST_FILE = (4, 1)
    _SIZE = 5

    @classmethod
    def write_fake(
        cls,
        handler: BinaryIO,
        name_offset: int,
        folder: Tuple[int, int],
        file: Tuple[int, int],
    ):
        writer = FakeSgaTocFolder(handler)
        writer.name_offset = name_offset
        writer.first_folder = folder[0]
        writer.last_folder = folder[1]
        writer.first_file = file[0]
        writer.last_file = file[1]


@pytest.mark.parametrize("name_offset", [0, 1])
@pytest.mark.parametrize("folder", [(1, 2), (2, 3)])
@pytest.mark.parametrize("file", [(4, 0), (0, 4)])
def test_fake_sga_toc_folder(
    name_offset: int, folder: Tuple[int, int], file: Tuple[int, int]
):
    with BytesIO() as handle:
        FakeSgaTocFolder.write_fake(handle, name_offset, folder, file)
        handle.seek(0)
        reader = FakeSgaTocFolder(handle)
        assert reader.name_offset == name_offset
        assert reader.first_folder == folder[0]
        assert reader.last_folder == folder[1]
        assert reader.first_file == file[0]
        assert reader.last_file == file[1]


@pytest.mark.parametrize(
    "version", [Version(major, minor) for (major, minor) in zip(range(2), range(2))]
)
def test_sga_file(version: Version):
    with BytesIO() as handle:
        MAGIC_WORD.write(handle)
        module.VersionSerializer.write(handle, version)
        handle.seek(0)
        reader = module.SgaFile(handle)
        assert reader.magic_word == MAGIC_WORD._expected
        assert reader.version == version


@pytest.mark.parametrize(
    "version", [Version(major, minor) for (major, minor) in zip(range(2), range(2))]
)
def test_sga_file_as_writer(version: Version):
    with BytesIO() as handle:
        MAGIC_WORD.write(handle)
        writer = module.SgaFile(handle)
        writer.version = version
        handle.seek(0)
        reader = module.SgaFile(handle)
        assert reader.magic_word == MAGIC_WORD._expected
        assert reader.version == version


def _generate_test_version_sample(v: Version):
    b1 = v.major.to_bytes(2, byteorder="little", signed=False)
    b2 = v.minor.to_bytes(2, byteorder="little", signed=False)
    b = bytearray(b1) + b2
    return {"buffer": b, "version": v}


TestVersionSamples = [
    _generate_test_version_sample(Version(major, minor))
    for (major, minor) in zip(range(2), range(2))
]


class TestVersionSerializer:
    @pytest.mark.parametrize(
        ["buffer", "expected"],
        [(d["buffer"], d["version"]) for d in TestVersionSamples],
    )
    def test_unpack(self, buffer: bytes, expected: Version):
        result = module.VersionSerializer.unpack(buffer)
        assert result == expected

    @pytest.mark.parametrize(
        ["buffer", "expected"],
        [(d["buffer"], d["version"]) for d in TestVersionSamples],
    )
    def test_read(self, buffer: bytes, expected: Version):
        with BytesIO(buffer) as stream:
            result = module.VersionSerializer.read(stream)
            assert result == expected

    @pytest.mark.parametrize(
        ["version", "expected"],
        [(d["version"], d["buffer"]) for d in TestVersionSamples],
    )
    def test_pack(self, version: Version, expected: bytes):
        result = module.VersionSerializer.pack(version)
        assert result == expected

    @pytest.mark.parametrize(
        ["version", "expected"],
        [(d["version"], d["buffer"]) for d in TestVersionSamples],
    )
    def test_write(self, version: Version, expected: bytes):
        with BytesIO() as stream:
            module.VersionSerializer.write(stream, version)

            result = stream.getvalue()
            assert result == expected


@pytest.mark.parametrize("len_mode", [True, False])
@pytest.mark.parametrize(
    "names",
    [
        ["alice"],
        ["alice", "bob"],
        ["alice", "bob", "charles"],
        [
            "a-really-long-name-that-exceeds-64-bytes-because-i-set-that-as-the-arbitrary-name-buffer-size"
        ],
    ],
)
@pytest.mark.parametrize("cacheable", [True, False, None])
def test_sga_name_window(names: List[str], len_mode: bool, cacheable: Optional[bool]):
    ENCODING = "utf-8"

    def generate_buffer_and_lookup():
        _lookup = {}
        with BytesIO() as stream:
            for _name in names:
                _lookup[stream.tell()] = _name
                _buffer = _name.encode(ENCODING) + b"\0"
                stream.write(_buffer)
            return stream.getvalue(), _lookup

    buffer, lookup = generate_buffer_and_lookup()
    count = len(names) if not len_mode else len(buffer)
    with BytesIO(buffer) as handle:
        window = module.SgaNameWindow(
            handle, 0, count, len_mode, encoding=ENCODING, cacheable=cacheable
        )
        for offset, name in lookup.items():
            result = window.get_name(offset)
            assert result == name


@pytest.mark.parametrize("len_mode", [True, False])
@pytest.mark.parametrize("cacheable", [True, False, None])
def test_sga_name_window_reinit_cache(len_mode: bool, cacheable: Optional[bool]):
    ENCODING = "utf-8"

    with BytesIO(b"") as handle:
        window = module.SgaNameWindow(
            handle, 0, 0, len_mode, encoding=ENCODING, cacheable=cacheable
        )
        cache = window._cache
        window._init_cache()
        new_cache = window._cache
        assert new_cache == cache


with BytesIO() as _h:
    FakeSgaTocDrive.write_fake(_h, "test", "fake", (0, 1), (2, 3), 4)
    _FAKE_SGA_TOC_DRIVE_BUFFER = _h.getvalue()
with BytesIO() as _h:
    FakeSgaTocFolder.write_fake(_h, 4, (0, 1), (2, 3))
    _FAKE_SGA_TOC_FOLDER_BUFFER = _h.getvalue()


class BadTocWindow: ...


def test_sga_toc_info_area_bad_window_size():
    try:
        with BytesIO() as h:
            area = module.SgaTocInfoArea(h, 0, 0, BadTocWindow)  # type: ignore
    except RelicToolError as e:
        pass
    else:
        pytest.fail("Expected a Relic Tool error!")


@pytest.mark.parametrize(
    ["cls", "buffer"],
    [
        (FakeSgaTocDrive, _FAKE_SGA_TOC_DRIVE_BUFFER),
        (FakeSgaTocFolder, _FAKE_SGA_TOC_FOLDER_BUFFER),
    ],
)
@pytest.mark.parametrize("specify_size", [True, False])
def test_init(buffer: bytes, cls: Type[module._TocWindowCls], specify_size: bool):
    with BytesIO(buffer) as r:
        cls_size = None
        if specify_size:
            if not hasattr(cls, "_SIZE"):
                pytest.skip("Cannot specify size; size not known")
            cls_size = cls._SIZE
        area = module.SgaTocInfoArea(r, 0, 1, cls, cls_size)
        assert len(area) == 1


@pytest.mark.parametrize(
    ["cls", "buffer"],
    [
        (FakeSgaTocDrive, _FAKE_SGA_TOC_DRIVE_BUFFER),
        (FakeSgaTocFolder, _FAKE_SGA_TOC_FOLDER_BUFFER),
    ],
)
def test_get_item_by_slice(buffer: bytes, cls: Type[module._TocWindowCls]):
    with BytesIO(buffer) as r:
        area = module.SgaTocInfoArea(r, 0, 1, cls)
        assert len(area) == 1
        sliced = area[0:1]
        assert sliced[0] == area[0]


@pytest.mark.parametrize(
    ["cls", "buffer"],
    [
        (FakeSgaTocDrive, _FAKE_SGA_TOC_DRIVE_BUFFER),
        (FakeSgaTocFolder, _FAKE_SGA_TOC_FOLDER_BUFFER),
    ],
)
def test_get_item(buffer: bytes, cls: Type[module._TocWindowCls]):
    with BytesIO(buffer) as r:
        area = module.SgaTocInfoArea(r, 0, 1, cls)
        _ = area[0]


@pytest.mark.parametrize(
    ["cls", "buffer"],
    [
        (FakeSgaTocDrive, _FAKE_SGA_TOC_DRIVE_BUFFER),
        (FakeSgaTocFolder, _FAKE_SGA_TOC_FOLDER_BUFFER),
    ],
)
def test_get_item_oob(buffer: bytes, cls: Type[module._TocWindowCls]):
    with BytesIO(buffer) as r:
        area = module.SgaTocInfoArea(r, 0, 1, cls)
        try:
            _ = area[1]
        except IndexError:
            pass
        else:
            pytest.fail("Should have raised IndexError")


@pytest.mark.parametrize(
    ["cls", "buffer"],
    [
        (FakeSgaTocDrive, _FAKE_SGA_TOC_DRIVE_BUFFER),
        (FakeSgaTocFolder, _FAKE_SGA_TOC_FOLDER_BUFFER),
    ],
)
def test_get_item(buffer: bytes, cls: Type[module._TocWindowCls]):
    with BytesIO(buffer) as r:
        area = module.SgaTocInfoArea(r, 0, 1, cls)
        for i, item in enumerate(area):
            assert item == area[i]
