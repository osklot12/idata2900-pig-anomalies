import random

from typing import List, Optional, Dict

from src.data.dataclasses.dataset_file import DatasetFile
from src.data.dataclasses.dataset_instance import DatasetInstance
from src.data.dataset.data_type import DataType
from src.data.dataset.providers.dataset_instance_provider import DatasetInstanceProvider
from src.data.dataset.sources.dataset_source import DatasetSource
from src.data.dataset.matching.matching_error_strategy import MatchingErrorStrategy
from src.data.dataset.matching.silent_removal_strategy import SilentRemovalStrategy


class DatasetFileMatcher(DatasetInstanceProvider):
    """A simple dataset entry provider, matching files by basename."""

    def __init__(self, source: DatasetSource, video_suffixes: List[str], annotation_suffixes: List[str]):
        """
        Initializes a DatasetFileMatcher instance.

        Args:
            source (DatasetSource): the dataset source
            video_suffixes (List[str]): list of video suffixes
            annotation_suffixes (List[str]): list of annotation suffixes
        """
        if source is None:
            raise ValueError("Source cannot be None")

        self._unmatched_files = source.get_source_ids()
        self._matched_pairs: List[DatasetInstance] = []

        self._suffixes: Dict[DataType, List[str]] = {
            DataType.VIDEO: video_suffixes,
            DataType.ANNOTATION: annotation_suffixes
        }

        self._match_error_strategy = SilentRemovalStrategy(self.remove_unmatched_file)

    def get_dataset_instance(self) -> Optional[DatasetInstance]:
        pair = None
        if random.random() < self._get_unmatched_ratio():
            pair = self._match_pair()

        if not pair and self._matched_pairs:
            pair = self._get_matched_pair()

        return pair

    def n_file_paths(self) -> int:
        """
        Returns the number of file paths currently in the matcher.

        Returns:
            int: the number of file paths in the matcher
        """
        return len(self._unmatched_files) + 2 * len(self._matched_pairs)

    def _match_pair(self) -> Optional[DatasetInstance]:
        """Matches a new pair from unmatched files."""
        result = None

        file = self._pick_random_valid_unmatched_file()
        if file is not None:
            match = self._find_match(file.file_path, self._suffixes[file.type.flip()])
            if match is None:
                self._match_error_strategy.handle_no_match(file.file_path)
            else:
                result = self._create_pair(file.file_path, match, file.type)
                self._handle_matched_pair(result)

        return result

    def _get_matched_pair(self) -> DatasetInstance:
        """Returns a random pair from the matched pairs."""
        if not self._matched_pairs:
            raise ValueError("No available pairs to sample from.")

        return random.choice(self._matched_pairs)

    def _get_unmatched_ratio(self):
        """Returns the of the unmatched files to all files."""
        n_files = len(self._unmatched_files) + 2 * len(self._matched_pairs)
        return len(self._unmatched_files) / n_files if n_files > 0 else 0.0

    def _pick_random_valid_unmatched_file(self) -> Optional[DatasetFile]:
        """Returns a random file from unmatched files with a valid extension, or None if not found."""
        file = None

        if self._unmatched_files:
            # set max attempts for each unique length of unmatched files
            attempts = 0
            max_attempts = 10
            last_n_files = len(self._unmatched_files)

            file = self._get_random_file()
            while file.type == DataType.UNKNOWN and attempts < max_attempts and self._unmatched_files:
                self._match_error_strategy.handle_unknown_file(file.file_path)

                if self._unmatched_files:
                    file = self._get_random_file()

                attempts += 1

                # reset attempts if a file has been removed
                current_n_files = len(self._unmatched_files)
                if current_n_files < last_n_files:
                    attempts = 0
                    last_n_files = current_n_files

        return file if file.type is not DataType.UNKNOWN else None

    def _get_random_file(self) -> DatasetFile:
        """Returns a random DatasetFile, or None if none are available."""
        file = random.choice(self._unmatched_files)
        data_type = self._which_data_type(file)
        return DatasetFile(file, data_type)

    @staticmethod
    def _create_pair(file_a: str, file_b: str, first_data_type: DataType) -> DatasetInstance:
        """Creates a DatasetFilePair from two file paths."""
        return DatasetInstance(file_a, file_b) if first_data_type == DataType.VIDEO else DatasetInstance(file_b, file_a)

    def _handle_matched_pair(self, pair: DatasetInstance):
        """Handles the matching of a new pair, moving data between internal structures."""
        self._unmatched_files.remove(pair.video_file)
        self._unmatched_files.remove(pair.annotation_file)
        self._matched_pairs.append(pair)

    def set_error_strategy(self, strategy: MatchingErrorStrategy) -> None:
        """
        Sets the strategy for handling errors while matching.

        Args:
            strategy (MatchingErrorStrategy): the strategy to use
        """
        self._error_strategy = strategy

    def remove_unmatched_file(self, file_name: str) -> None:
        """
        Removes a file path from the list of unmatched files.
        Does not remove the file from the source.

        Args:
            file_name (str): name of file to remove
        """
        if file_name in self._unmatched_files:
            self._unmatched_files.remove(file_name)