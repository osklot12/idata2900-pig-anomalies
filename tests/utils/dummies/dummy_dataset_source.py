from typing import List

from src.data.dataset.dataset_source import DatasetSource


class DummyDatasetSource(DatasetSource):
    """A dummy DatasetSource implementation for testing."""

    def __init__(self):
        # default file paths
        self.file_paths: List[str] = [
            "dir_a/dir_b/id1.mp4",
            "dir_a/dir_b/id2.mp4",
            "dir_a/dir_b/id3.mp4",
            "dir_a/dir_c/id1.json",
            "dir_a/dir_c/id2.json",
            "dir_a/dir_c/id3.json",
        ]

    def set_file_paths(self, file_paths: List[str]):
        """
        Sets custom file paths to provide.

        Args:
            file_paths (List[str]): list of file paths
        """
        self.file_paths = file_paths

    def list_files(self) -> List[str]:
        return self.file_paths