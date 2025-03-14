from typing import Optional

from src.data.dataclasses.dataset_file_pair import DatasetFilePair
from src.data.dataset.dataset_file_pair_provider import DatasetFilePairProvider
from src.data.dataset.dataset_source import DatasetSource
from src.data.dataset.matching.matching_strategy import MatchingStrategy
from src.data.dataset.selection.file_selection_strategy import FileSelectionStrategy


class SimpleFilePairProvider(DatasetFilePairProvider):
    """Returns randomly picked dataset file pair instances."""

    def __init__(self, source: DatasetSource, video_selector: FileSelectionStrategy, annotation_matcher: MatchingStrategy):
        """
        Initializes a RandomFilePairProvider instance.

        Args:
            source (DatasetSource): a dataset source
            video_selector (FileSelectionStrategy): a strategy for selecting video files
            annotation_matcher (MatchingStrategy): a strategy for finding an annotation for the video file
        """
        self._source = source
        self._video_selector = video_selector
        self._matcher = annotation_matcher

    def get_file_pair(self) -> Optional[DatasetFilePair]:
        result = None

        files = self._source.list_files()
        if files:
            video_file = self._video_selector.select_file(files)
            if video_file:
                annotation_file = self._matcher.find_match(video_file, files)
                if annotation_file:
                    result = DatasetFilePair(video_file, annotation_file)

        return result