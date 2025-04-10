from typing import Tuple

from src.data.dataset.registries.factories.file_registry_factory import FileRegistryFactory
from src.data.dataset.registries.file_registry import FileRegistry
from src.data.dataset.registries.suffix_file_registry import SuffixFileRegistry


class SuffixFileRegistryFactory(FileRegistryFactory):
    """Factory for creating SuffixFileRegistry instances."""

    def __init__(self, source_factory: FileRegistryFactory, suffixes: Tuple[str, ...]):
        """
        Initializes a SuffixFileRegistryFactory instance.

        Args:
            source_factory (FileRegistryFactory): factory for creating source file registries
            suffixes (Tuple[str, ...]): tuple of suffix strings
        """
        self._source_factory = source_factory
        self._suffixes = suffixes

    def create_registry(self) -> FileRegistry:
        return SuffixFileRegistry(
            source=self._source_factory.create_registry(),
            suffixes=self._suffixes
        )