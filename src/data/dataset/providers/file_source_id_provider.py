from typing import List

from src.data.dataset.providers.source_ids_provider import SourceIDsProvider
from src.data.dataset.sources.dataset_source import DatasetSource
from src.data.parsing.string_parser import StringParser


class FileSourceIDsProvider(SourceIDsProvider):
    """A provider of dataset file source IDs using string parsing."""

    def __init__(self, dataset_source: DatasetSource, string_parser: StringParser):
        """
        Initializes a DatasetSourceIDsProvider.

        Args:
            dataset_source (DatasetSource): the dataset source
            string_parser (StringParser): the string parser for parsing source IDs
        """
        self._dataset_source = dataset_source
        self._string_parser = string_parser

    def get_source_ids(self) -> List[str]:
        return [self._string_parser.parse_string(f) for f in self._dataset_source.list_files()]