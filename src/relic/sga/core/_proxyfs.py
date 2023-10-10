from __future__ import annotations

from typing import BinaryIO, List, Iterable

from relic.sga.core import StorageType
from relic.sga.core.lazyio import BinaryWindow, ZLibFileReader
from relic.sga.core.serialization import (
    SgaTocFile,
    SgaNameWindow,
    SgaTocFolder,
    SgaTocDrive,
    SgaToc,
    SgaFile,
    SgaMetaBlock,
)


class SgaFsFile:
    def __init__(
        self, info: SgaTocFile, name_window: SgaNameWindow, data_window: BinaryWindow
    ):
        self._info = info
        self._name_window = name_window
        self._data_window = data_window

    @property
    def name(self):
        return self._name_window.get_name(self._info.name_offset)

    def data(self, decompress: bool = True) -> BinaryIO:
        comp_size = self._info.compressed_size
        decomp_size = self._info.decompressed_size
        window = BinaryWindow(
            self._data_window, self._info.data_offset, self._info.compressed_size
        )
        if (
            decompress
            and self._info.storage_type != StorageType.STORE
            and comp_size != decomp_size
        ):
            return ZLibFileReader(window)
        else:
            return window

    def verify(self) -> bool:
        if hasattr(self._info, "verify"):
            return self._info.verify(self)
        else:
            raise NotImplementedError


class SgaFsFolder:
    def __init__(
        self,
        info: SgaTocFolder,
        name_window: SgaNameWindow,
        files: List[SgaFsFile],
        folders: List["SgaFsFolder"],
    ):
        self._info = info
        self._name_window = name_window
        self._files = files
        self._folders = folders

    @property
    def files(self) -> List[SgaFsFile]:
        start, stop = self._info.first_file, self._info.last_file
        return self._files[start:stop]

    @property
    def folders(self) -> List["SgaFsFolder"]:
        start, stop = self._info.first_folder, self._info.last_folder
        return self._folders[start:stop]

    @property
    def name(self):
        return self._name_window.get_name(self._info.name_offset)

    @property
    def iter_folders(self) -> Iterable["SgaFsFolder"]:
        return iter(self.folders)

    @property
    def iter_files(self) -> Iterable[SgaFsFile]:
        return iter(self.files)

    def walk(self):
        folders = list(self.iter_folders)
        files = list(self.iter_files)
        yield self, folders, files
        for folder in folders:
            yield from folder.walk()


class LazySgaFsFolder(SgaFsFolder):
    def __init__(
        self,
        info: SgaTocFolder,
        name_window: SgaNameWindow,
        data_window: BinaryWindow,
        files: List[SgaTocFile],
        folders: List[SgaTocFolder],
    ):
        super().__init__(info, name_window, None, None)
        self._data_window = data_window
        self._files_info = files
        self._folders_info = folders

    @property
    def files(self) -> List[SgaFsFile]:
        if self._files is None:
            info = self._info
            sub_files = self._files[info.first_file : info.last_file]
            self._files = [
                SgaFsFile(file, self._name_window, self._data_window)
                for file in sub_files
            ]
        return self._files

    @property
    def folders(self) -> List[SgaFsFolder]:
        if self._folders is None:
            info = self._info
            sub_folders = self._folders_info[info.first_folder : info.last_folder]
            self._folders = [
                LazySgaFsFolder(folder, self._name_window, self._data_window)
                for folder in sub_folders
            ]
        return self._folders


class SgaFsDrive:
    def __init__(
        self,
        info: SgaTocDrive,
        all_files: List[SgaFsFile],
        all_folders: List[SgaFsFolder],
    ):
        self._info = info
        self._files = all_files
        self._folders = all_folders

    @property
    def files(self) -> List[SgaFsFile]:
        start, stop = self._info.first_file, self._info.last_file
        return self._files[start:stop]

    @property
    def folders(self) -> List[SgaFsFolder]:
        start, stop = self._info.first_folder, self._info.last_folder
        return self._folders[start:stop]

    @property
    def name(self):
        return self._info.name

    @property
    def path(self):
        return self._info.path

    @property
    def iter_folders(self) -> Iterable[SgaFsFolder]:
        return iter(self.folders)

    @property
    def iter_files(self) -> Iterable[SgaFsFile]:
        return iter(self.files)

    @property
    def root(self) -> SgaFsFolder:
        return self._folders[self._info.root_folder]

    def walk(self):
        # root = self.root
        # name = root.name
        # folders = list(root.iter_folders)
        # files = list(root.iter_files)
        # __ = (self.name, self.path),
        # if name != "":
        #     raise NotImplementedError(__, name)
        # if len(folders) != 1:
        #     raise NotImplementedError(__, len(folders), [f.name for f in folders])
        # if len(files) != 0:
        #     raise NotImplementedError(__, len(files), [f.name for f in files])

        yield from self.root.walk()


class LazySgaFsDrive(SgaFsDrive):
    def __init__(self, info: SgaTocDrive, toc: SgaToc, data_window: BinaryWindow):
        super().__init__(info, None, None)
        self._root = None
        self._toc = toc
        self._data_window = data_window

    @property
    def files(self) -> List[SgaFsFile]:
        raise NotImplementedError

    @property
    def folders(self) -> List[SgaFsFolder]:
        raise NotImplementedError

    @property
    def root(self) -> SgaFsFolder:
        if self._root is None:
            info = self._info
            drive_folders: List[SgaTocFolder] = self._toc.folders[
                info.first_folder : info.last_folder
            ]
            drive_files: List[SgaTocFile] = self._toc.files[
                info.first_file : info.last_file
            ]
            drive_root = self._toc.folders[info.root_folder]
            self._root = LazySgaFsFolder(
                drive_root,
                self._toc.names,
                self._data_window,
                drive_files,
                drive_folders,
            )
        return self._root


class SgaFs:
    def __init__(self, file: SgaFile):
        self._stream = file._parent
        self._file_handler = file
        self._files = None
        self._folders = None
        self._drives = None
        self.__post_init()

    def __post_init(self):
        file = self._file_handler
        toc = file.table_of_contents
        data_window = file.data_block
        self._files = [
            SgaFsFile(f_info, toc.names, data_window) for f_info in toc.files
        ]
        self._folders = []
        for dir_info in toc.folders:
            self._folders.append(
                SgaFsFolder(dir_info, toc.names, self._files, self._folders)
            )
        self._drives = [
            SgaFsDrive(drive_info, self._files, self._folders)
            for drive_info in toc.drives
        ]

    @property
    def meta(self) -> SgaMetaBlock:
        return self._file_handler.meta

    @property
    def drives(self) -> List[SgaFsDrive]:
        return self._drives

    def walk(self):
        for drive in self.drives:
            yield drive, drive.walk()

    def verify(self) -> bool:
        """
        Verifies that the Archive is not corrupted.

        SgaFSFile.verify() should be used to verify individual files
        """

        if hasattr(self._file_handler, "verify"):
            return self._file_handler.verify()
        else:
            raise NotImplementedError


class LazySgaFs(SgaFs):
    def __init__(self, file: SgaFile):
        super().__init__(file)

    @property
    def drives(self) -> List[SgaFsDrive]:
        if self._drives is None:
            file = self._file_handler
            toc = file.table_of_contents
            data_window = file.data_block
            self._drives = [
                LazySgaFsDrive(drive_info, toc, data_window)
                for drive_info in toc.drives
            ]
        return super().drives

    def __post_init(self):
        pass  # Do nothing
