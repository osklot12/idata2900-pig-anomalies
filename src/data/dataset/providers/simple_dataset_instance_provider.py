from typing import Optional

from src.data.dataclasses.dataset_instance import DatasetInstance
from src.data.dataset.providers.dataset_catalog import DatasetManifest
from src.data.dataset.sources.dataset_source_registry import SourceRegistry
from src.data.dataset.matching.matching_strategy import MatchingStrategy
from src.data.dataset.selection.file_selection_strategy import FileSelectionStrategy


class SimpleDatasetInstanceProvider(DatasetManifest):
    """Returns randomly picked dataset instances."""

    def __init__(self, source: SourceRegistry, video_selector: FileSelectionStrategy, annotation_matcher: MatchingStrategy):
        """
        Initializes a SimpleDatasetInstanceProvider instance.

        Args:
            source (SourceRegistry): a dataset source
            video_selector (FileSelectionStrategy): a strategy for selecting video files
            annotation_matcher (MatchingStrategy): a strategy for finding an annotation for the video file
        """
        self._source = source
        self._video_selector = video_selector
        self._matcher = annotation_matcher

    def get_dataset_instance(self) -> Optional[DatasetInstance]:
        result = None

        files = self._source.get_source_ids()
        if files:

            searching = True
            while searching:
                video_file = self._video_selector.select_file(files)

                if video_file:
                    annotation_file = self._matcher.find_match(video_file, files)

                    if annotation_file:
                        result = DatasetInstance(video_file, annotation_file)
                        searching = False
                    else:
                        files.remove(video_file)
                else:
                    searching = False

        return result