from typing import List

from src.data.dataclasses.dataset_instance import DatasetInstance
from src.data.dataset.manifest.manifest import Manifest
from src.data.dataset.matching.matching_strategy import MatchingStrategy
from src.data.dataset.registries.file_registry import FileRegistry


class MatchingManifest(Manifest):
    """Dataset manifest that creates instances by matching."""

    def __init__(self, video_registry: FileRegistry, annotations_registry: FileRegistry, matcher: MatchingStrategy):

    def list_all_ids(self) -> List[str]:
        pass

    def get_instance(self, instance_id: str) -> DatasetInstance:
        pass

