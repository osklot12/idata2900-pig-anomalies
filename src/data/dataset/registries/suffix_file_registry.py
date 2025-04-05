from typing import List, Tuple

from src.data.dataset.registries.file_registry import FileRegistry


class SuffixFileRegistry(FileRegistry):
    """Registry that filters out files by suffixes."""

    def __init__(self, source: FileRegistry, suffixes: Tuple[str, ...]):
        """
        Initializes a VideoFileRegistry instance.

        Args:
            source (FileRegistry): the file registry to use
            suffixes (List[str]): the suffixes that the registry allows
        """
        self._source = source
        self._suffixes = suffixes

    def get_file_paths(self) -> set[str]:
        return {path for path in self._source.get_file_paths() if path.endswith(self._suffixes)}