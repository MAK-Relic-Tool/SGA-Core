from __future__ import annotations

import os
from os.path import expanduser
from typing import (
    Dict,
    Optional,
    Protocol,
    BinaryIO,
    TypeVar,
    List,
    Iterable,
    Union,
    Type,
)

import fs.opener
from fs.opener import Opener
from fs.opener.errors import OpenerError
from fs.opener.parse import ParseResult
from relic.core.errors import RelicToolError
from relic.core.lazyio import BinaryProxy, get_proxy
from relic.core.entrytools import EntrypointRegistry

from relic.sga.core.definitions import Version, MAGIC_WORD
from relic.sga.core.essencefs.definitions import EssenceFS
from relic.sga.core.serialization import (
    VersionSerializer,
)

_TEssenceFS = TypeVar("_TEssenceFS", bound=EssenceFS)


# Reimplement Opener as a Typed-Protocol # Ugly, but it's my ugly
#   This should allow it to be used as an opener; or as a plugin-only opener for the EssenceFSOpener
class EssenceFsOpenerPlugin(Protocol[_TEssenceFS]):  # type: ignore
    @property
    def protocols(self) -> List[str]:
        raise NotImplementedError

    @property
    def versions(self) -> List[Version]:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    def open_fs(
        self,
        fs_url: str,
        parse_result: ParseResult,
        writeable: bool,
        create: bool,
        cwd: str,
    ) -> _TEssenceFS:
        raise NotImplementedError


def _get_version(file: Union[BinaryProxy, BinaryIO], advance: bool = False) -> Version:
    binio = get_proxy(file)
    start = binio.tell()
    MAGIC_WORD.validate(binio, advance=True)
    version = VersionSerializer.read(binio)
    if not advance:
        binio.seek(start, os.SEEK_CUR)
    return version


class EssenceFsOpener(
    EntrypointRegistry[Version, EssenceFsOpenerPlugin[_TEssenceFS]], Opener
):
    EP_GROUP = "relic.sga.opener"

    protocols = ["sga"]

    def __init__(
        self,
        # data: Optional[Dict[Version, EssenceFsOpenerPlugin]] = None,
        autoload: bool = True,
    ):
        super().__init__(
            entry_point_path=self.EP_GROUP,
            key_func=self._version2key,  # type: ignore # WHY?
            auto_key_func=self._val2keys,
            # data=data,
            autoload=autoload,
        )

    @staticmethod
    def _version2key(version: Version) -> str:
        return f"v{version.major}.{version.minor}"

    @staticmethod
    def _value2keys(plugin: EssenceFsOpenerPlugin[_TEssenceFS]) -> Iterable[Version]:
        yield from plugin.versions

    def open_fs(
        self,
        fs_url: str,
        parse_result: ParseResult,
        writeable: bool,
        create: bool,
        cwd: str,
    ) -> EssenceFS:
        # All EssenceFS should be writable; so we can ignore that

        if parse_result.resource == "":
            if create:
                raise RelicToolError(
                    "Cannot create an SGA from fs.open_fs or relic.sga.core.essencefs.open_sga;"
                    " please manually create an empty FS object from an appropriate SGA Plugin."
                )
            raise fs.opener.errors.OpenerError(
                "No path was given and opener not marked for 'create'!"
            )

        path = os.path.abspath(os.path.join(cwd, expanduser(parse_result.resource)))
        with open(path, "rb") as peeker:
            version = _get_version(
                peeker, True
            )  # advance is true to avoid unnecessary seek
        try:
            opener: Union[Type[EssenceFsOpenerPlugin], EssenceFsOpenerPlugin] = self[version]  # type: ignore
        except KeyError as e:
            raise RelicToolError(
                f"Version {version} not supported! Supported SGA Versions '{list(self.keys())}'."
            )

        if isinstance(opener, type):
            opener: EssenceFsOpenerPlugin = opener()  # type: ignore

        return opener.open_fs(fs_url, parse_result, writeable, create, cwd)  # type: ignore


registry: EssenceFsOpener[EssenceFS] = EssenceFsOpener()


class _EssenceFsOpenerAdapter(Opener):
    """
    An adapter allowing PyFileSystem to use the EssenceFS Registry

    Required for PyFilesystem to use non-entrypoint plugins
    (I.E. Plugins registered manually)

    PyFileSystem expects a subclass of Opener, not an instance of Opener
    This adapter allows PyFileSystem to create a new opener, while using the global opener instance under the hood
    """

    @classmethod
    @property
    def protocols(cls) -> List[str]:
        return registry.protocols

    def open_fs(self, fs_url, parse_result, writeable, create, cwd):
        return registry.open_fs(fs_url, parse_result, writeable, create, cwd)


open_sga = registry.open_fs
