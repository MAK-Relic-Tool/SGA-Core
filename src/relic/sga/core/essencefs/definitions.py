from __future__ import annotations

from abc import ABC
from typing import Iterator, Tuple, Any, Dict

from fs.base import FS


class EssenceFS(FS, ABC):
    def iterate_fs(self) -> Iterator[Tuple[str, FS]]:
        raise NotImplementedError

    def info_tree(self, **options: Any) -> Dict[str, Any]:
        """
        Get a dictionary of the Filesystem tree, containing metadata for files/folders, 'drives' and the root archive

        :rtype: Dict[str,Any]
        :returns: A dictionary representing the file system tree and it's metadata
        """
        raise NotImplementedError
