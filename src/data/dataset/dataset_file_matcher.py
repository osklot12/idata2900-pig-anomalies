import random
import os

from typing import List, Optional, Dict

from src.data.dataclasses.dataset_file import DatasetFile
from src.data.dataclasses.dataset_file_pair import DatasetFilePair
from src.data.dataset.data_type import DataType
from src.data.dataset.dataset_entry_provider import DatasetEntryProvider
from src.data.dataset.dataset_source import DatasetSource
from src.data.dataset.matching_error_strategy import MatchingErrorStrategy
from src.data.dataset.silent_removal_strategy import SilentRemovalStrategy


class DatasetFileMatcher(DatasetEntryProvider):
    """A simple dataset entry provider, matching entries based on their suffixes."""

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

        self._unmatched_files = source.list_files()
        self._matched_pairs: List[DatasetFilePair] = []

        self._suffixes: Dict[DataType, List[str]] = {
            DataType.VIDEO: video_suffixes,
            DataType.ANNOTATION: annotation_suffixes
        }

        self._match_error_strategy = SilentRemovalStrategy(self.remove_unmatched_file)

    def get_random(self) -> Optional[DatasetFilePair]:
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

    def _match_pair(self) -> Optional[DatasetFilePair]:
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

    def _get_matched_pair(self) -> DatasetFilePair:
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
    def _create_pair(file_a: str, file_b: str, first_data_type: DataType) -> DatasetFilePair:
        """Creates a DatasetFilePair from two file paths."""
        return DatasetFilePair(file_a, file_b) if first_data_type == DataType.VIDEO else DatasetFilePair(file_b, file_a)

    def _handle_matched_pair(self, pair: DatasetFilePair):
        """Handles the matching of a new pair, moving data between internal structures."""
        self._unmatched_files.remove(pair.video_file)
        self._unmatched_files.remove(pair.annotation_file)
        self._matched_pairs.append(pair)

    def _find_match(self, file_name: str, target_suffixes: List[str]) -> Optional[str]:
        """Tries to find a file name matching the given file name, but with a suffix from target_suffixes."""
        base_name = os.path.splitext(os.path.basename(file_name))[0]

        return next(
            (
                match_path for match_path in self._unmatched_files
                if os.path.splitext(os.path.basename(match_path))[0] == base_name
                and any(match_path.endswith(suffix) for suffix in target_suffixes)
            ), None
        )

    def _which_data_type(self, filename: str) -> DataType:
        """Returns the DataType corresponding to the filename suffix, or DataType.UNKNOWN if unknown."""
        return next(
            (
                data_type for data_type, suffixes in self._suffixes.items()
                if any(filename.endswith(suffix) for suffix in suffixes)
            ), DataType.UNKNOWN
        )

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