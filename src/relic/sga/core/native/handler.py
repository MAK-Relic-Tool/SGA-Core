import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import TypeVar, Generic, Type, Iterable, Sequence, List

from relic.core.entrytools import EntrypointRegistry
from relic.core.errors import MismatchError, RelicToolError

from relic.sga.core.definitions import Version, MAGIC_WORD, StorageType
from relic.sga.core.errors import VersionNotSupportedError, VersionMismatchError
from relic.sga.core.native.definitions import ReadonlyMemMapFile, FileEntry, Result

_TFileEntry = TypeVar("_TFileEntry", bound=FileEntry)


class SgaReader(ReadonlyMemMapFile):
    """
    Reads an SGA file using FileEntry objects.

    FileEntry objects contain ABSOLUTE byte offsets within the SGA file
    as well as the file's size (compressed_size)
    and the file's storage_type

    These three keys allow us to quickly read raw sga files and decompress them if needed
    """

    def read(self, offset: int, size: int) -> bytes:
        buffer = self._mmap_handle[offset : offset + size]
        if len(buffer) != size:
            raise MismatchError("read mismatch", len(buffer), size)
        return buffer

    def read_range(self, offset: int, terminal: int) -> bytes:
        return self.read(offset, terminal - offset)

    def read_file(self, entry: FileEntry, decompress: bool = True) -> bytes:
        raw = self.read(entry.data_offset, entry.compressed_size)
        if not decompress:
            return raw
        if entry.storage_type in [StorageType.STORE]:
            return raw
        if entry.storage_type in [
            StorageType.BUFFER_COMPRESS,
            StorageType.STREAM_COMPRESS,
        ]:
            zlib_buffer = zlib.decompress(raw)
            if len(zlib_buffer) != entry.decompressed_size:
                raise MismatchError(
                    "size mismatch", len(zlib_buffer), entry.decompressed_size
                )
            return zlib_buffer
        raise RelicToolError(
            f"read_buffer does not support StorageType='{entry.storage_type}'"
        )

    def open_file(self, entry: FileEntry, decompress: bool = True) -> BytesIO:
        return BytesIO(self.read_file(entry, decompress))

    def read_files_parallel(
        self, file_paths: Sequence[FileEntry], num_workers: int
    ) -> List[Result[FileEntry, bytes]]:
        """Read and decompress files in PARALLEL."""

        def read_decompress(entry: FileEntry) -> Result[FileEntry, bytes]:
            try:
                data = self.read_file(entry)
                return Result(entry, data)
            except Exception as e:
                return Result.create_error(entry, e)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(read_decompress, file_paths))

        return results


class NativeParserHandler(ReadonlyMemMapFile, Generic[_TFileEntry]):
    def parse(self) -> list[_TFileEntry]:
        raise NotImplementedError()

    def get_file_entries(self) -> list[_TFileEntry]:
        raise NotImplementedError()

    def get_drive_count(self) -> int:
        raise NotImplementedError()


class SharedHeaderParser(ReadonlyMemMapFile):
    def __init__(self, path: str):
        super().__init__(path)

    def read_magic(self) -> bytes:
        return self._read_range(0, 8)

    def check_magic(self) -> bool:
        buffer = self.read_magic()
        # TODO; make magicword accept bytes/bytearray
        return MAGIC_WORD.check(BytesIO(buffer))

    def validate_magic(self) -> None:
        buffer = self.read_magic()
        return MAGIC_WORD.validate(BytesIO(buffer))

    def read_version(self) -> Version:
        # Read version
        buffer = self._read_range(8, 12)
        major, minor = struct.unpack("<HH", buffer)
        version = Version(major, minor)
        return version

    def check_version(self, *versions: Version) -> bool:
        version = self.read_version()
        return version in versions

    def validate_version(self, *versions: Version) -> None:
        version = self.read_version()
        if version not in versions:
            if len(versions) == 1:
                raise VersionMismatchError(version, versions[0])
            else:
                raise VersionNotSupportedError(version, versions)


class NativeParserRegistry(
    EntrypointRegistry[Version, Type[NativeParserHandler[FileEntry]]]
):
    EP_GROUP = "relic.sga.parser"

    def __init__(
        self,
        autoload: bool = True,
    ):
        super().__init__(
            entry_point_path=self.EP_GROUP,
            key_func=self._version2key,  # type: ignore # WHY?
            auto_key_func=self._val2keys,
            autoload=autoload,
        )

    @staticmethod
    def _version2key(version: Version) -> str:
        return f"v{version.major}.{version.minor}"

    @staticmethod
    def _value2keys(
        plugin: Type[NativeParserHandler[_TFileEntry]],
    ) -> Iterable[Version]:
        raise NotImplementedError()

    def create_parser(self, path: str) -> NativeParserHandler[FileEntry]:
        with SharedHeaderParser(path) as shared_parser:
            shared_parser.validate_magic()
            version = shared_parser.read_version()
            try:
                parser_klass = self[version]
            except KeyError as e:

                def _key2version(key: str) -> Version:
                    major, minor = key.split(".")
                    return Version(int(major[1:]), int(minor))

                supported_versions = [_key2version(key) for key in self._backing.keys()]
                raise VersionNotSupportedError(version, supported_versions) from e
            return parser_klass(path)


parser_registry: NativeParserRegistry = NativeParserRegistry()
