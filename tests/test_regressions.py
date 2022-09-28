"""
Tests which ensures releases do not break backwards-compatibility by failing to expose modules/names
"""

import importlib
from typing import List, Iterable, Tuple

import pytest

core__all__ = [
    "definitions",
    "errors",
    "filesystem",
    "protocols",
    "serialization",
]


@pytest.mark.parametrize("submodule", core__all__)
def test_import_module(submodule: str):
    try:
        importlib.import_module(f"relic.sga.core.{submodule}")
    except ImportError:
        raise AssertionError(f"{submodule} is no longer exposed!")


definitions__all__ = ["MagicWord", "Version", "StorageType", "VerificationType"]
errors__all__ = [
    "VersionMismatchError",
    "MD5MismatchError",
    "VersionNotSupportedError",
    "DecompressedSizeMismatch",
]
fs__all__ = [
    "ESSENCE_NAMESPACE",
    "EssenceFSHandler",
    "EssenceFSFactory",
    "_EssenceFile",
    "_EssenceDirEntry",
    "_EssenceDriveFS",
    "EssenceFS",
    "registry",
    "EssenceFSOpener",
]
protocols__all__ = ["T", "StreamSerializer"]
serialization__all__ = [
    "TocBlock",
    "TocHeaderSerializer",
    "DriveDef",
    "DriveDefSerializer",
    "FolderDef",
    "FolderDefSerializer",
    "MetaBlock",
    "TMetaBlock",
    "TTocMetaBlock",
    "FileDef",
    "TFileDef",
    "AssembleFileMetaFunc",
    "DisassembleFileMetaFunc",
    "AssembleMetaFunc",
    "DisassembleMetaFunc",
    "TOCSerializationInfo",
    "FSAssembler",
    "FSDisassembler",
    "Md5ChecksumHelper",
    "EssenceFSSerializer",
    "FileLazyInfo",
    "ArchivePtrs",
]


def module_imports_helper(submodule: str, all: List[str]) -> Iterable[Tuple[str, str]]:
    return zip([submodule] * len(all), all)


@pytest.mark.parametrize(
    ["submodule", "attribute"],
    [
        *module_imports_helper("errors", errors__all__),
        *module_imports_helper("definitions", definitions__all__),
        *module_imports_helper("filesystem", fs__all__),
        *module_imports_helper("protocols", protocols__all__),
        *module_imports_helper("serialization", serialization__all__),
    ],
)
def test_module_imports(submodule: str, attribute: str):
    module = importlib.import_module(f"relic.sga.core.{submodule}")
    _ = getattr(module, attribute)
