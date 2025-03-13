import random
import os

from typing import Tuple, List, Optional, Dict

from src.data.dataset.data_type import DataType
from src.data.dataset.dataset_entry_provider import DatasetEntryProvider
from src.data.dataset.dataset_source import DatasetSource
from src.data.dataset.matching_error_strategy import MatchingErrorStrategy
from src.data.dataset.silent_removal_strategy import SilentRemovalStrategy


class SimpleDatasetEntryProvider(DatasetEntryProvider):
    """A simple dataset entry provider, holding all paths in memory."""

    def __init__(self, source: DatasetSource, video_suffixes: List[str], annotation_suffixes: List[str]):
        """
        Initializes a SimpleDatasetEntryProvider instance.

        Args:
            source (DatasetSource): the dataset source
            video_suffixes (List[str]): list of video suffixes
            annotation_suffixes (List[str]): list of annotation suffixes
        """
        if source is None:
            raise ValueError("Source cannot be None")

        self._unmatched_files = source.list_files()
        self._matched_pairs: List[Tuple[str, str]] = []

        self._suffixes: Dict[DataType, List[str]] = {
            DataType.VIDEO: video_suffixes,
            DataType.ANNOTATION: annotation_suffixes
        }

        self._match_error_strategy = SilentRemovalStrategy(self.remove_unmatched_file)

    def get_random(self) -> Tuple[str, str]:
        if not self._unmatched_files and not self._matched_pairs:
            raise ValueError("No available pairs to sample from.")

        n_files = len(self._unmatched_files) + 2 * len(self._matched_pairs)
        unmatched_ratio = len(self._unmatched_files) / n_files

        pair = None
        if random.random() < unmatched_ratio:
            pair = self._match_pair()

        if not pair and self._matched_pairs:
            pair = self._get_matched_pair()

        if not pair:
            raise RuntimeError("Could not find any pairs")

        return pair

    def _match_pair(self) -> Tuple[str, str]:
        """Matches a new pair from unmatched files."""
        if not self._unmatched_files:
            raise ValueError("No available files to sample from")

        pair = None

        attempts = 0
        max_attempts = 10
        last_n_paths = len(self._unmatched_files)
        while self._unmatched_files and not pair and attempts < max_attempts:
            file = random.choice(self._unmatched_files)
            pair = self._create_pair_for_file(file)

            if pair:
                self._handle_matched_pair(pair)
            else:
                attempts += 1

            if last_n_paths > len(self._unmatched_files):
                last_n_paths = len(self._unmatched_files)
                attempts = 0

        return pair

    def _get_matched_pair(self) -> Tuple[str, str]:
        """Returns a random pair from the matched pairs."""
        if not self._matched_pairs:
            raise ValueError("No available pairs to sample from.")

        return random.choice(self._matched_pairs)

    def _create_pair_for_file(self, file) -> Optional[Tuple[str, str]]:
        """Creates a pair for the given file, or None if no match was found."""
        result = None

        data_type = self._get_data_type(file)
        if data_type in self._suffixes:

            match = self._find_match(file, self._suffixes[data_type])
            if match:
                result = self._order_pair(file, match, data_type)
            else:
                self._match_error_strategy.handle_no_match(file)

        else:
            self._match_error_strategy.handle_unknown_file(file)

        return result

    @staticmethod
    def _order_pair(file_a: str, file_b: str, first_data_type: DataType) -> Tuple[str, str]:
        """Orders a pair of video-annotation file paths as (video, annotation)."""
        return (file_a, file_b) if first_data_type == DataType.VIDEO else (file_b, file_a)

    def _get_data_type(self, file_name: str) -> DataType:
        """Returns the data type for the given file."""
        result = DataType.UNKNOWN

        if self._is_data_type(file_name, DataType.VIDEO):
            result = DataType.VIDEO
        elif self._is_data_type(file_name, DataType.ANNOTATION):
            result = DataType.ANNOTATION

        return result

    def _handle_matched_pair(self, pair):
        """Handles the matching of a new pair, moving data between internal structures."""
        self._unmatched_files.remove(pair[0])
        self._unmatched_files.remove(pair[1])
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

    def _is_data_type(self, file: str, data_type: DataType) -> bool:
        """Checks if the given file has the given data type."""
        file_name = os.path.basename(file)
        return any(file_name.endswith(suffix) for suffix in self._suffixes.get(data_type, []))

    def remove_unmatched_file(self, file_name: str) -> None:
        """
        Removes a file path from the list of unmatched files.
        Does not remove the file from the source.

        Args:
            file_name (str): name of file to remove
        """
        if file_name in self._unmatched_files:
            self._unmatched_files.remove(file_name)