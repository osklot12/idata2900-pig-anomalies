from typing import List, Dict, Optional

from charset_normalizer.md import annotations

from src.data.dataclasses.dataset_instance import DatasetInstance
from src.data.dataset.identifiers.base_name_identifier import BaseNameIdentifier
from src.data.dataset.identifiers.identifier import Identifier
from src.data.dataset.manifests.manifest import Manifest
from src.data.dataset.matching.base_name_matcher import BaseNameMatcher
from src.data.dataset.matching.matching_strategy import MatchingStrategy
from src.data.dataset.registries.file_registry import FileRegistry


class MatchingManifest(Manifest):
    """Dataset manifests that creates instances by matching."""

    def __init__(self, video_registry: FileRegistry, annotations_registry: FileRegistry,
                 matcher: MatchingStrategy = BaseNameMatcher(), identifier: Identifier = BaseNameIdentifier()):
        """
        Initializes a MatchingManifest instance.

        Args:
            video_registry (FileRegistry): the video registry
            annotations_registry (FileRegistry): the annotations registry
            matcher (MatchingStrategy): the matching strategy for matching annotations with videos
            identifier (Identifier): identifier to assign IDs to the matched instances
        """
        self._video_registry = video_registry
        self._annotations_registry = annotations_registry
        self._matcher = matcher
        self._identifier = identifier

        self._instances: Dict[str, DatasetInstance] = {}

    @property
    def ids(self) -> List[str]:
        if not self._instances:
            self.update()
        return list(self._instances.keys())

    def get_instance(self, instance_id: str) -> Optional[DatasetInstance]:
        if not self._instances:
            self.update()
        return self._instances.get(instance_id, None)

    def update(self) -> None:
        """Matches video and annotation files and creates dataset instances."""
        video_files = self._video_registry.get_file_paths()
        annotations_files = self._annotations_registry.get_file_paths()

        for video in video_files:
            annotations_path = self._matcher.match(video, annotations_files)
            if annotations_path:
                self._instances[self._identifier.identify(video, annotations_path)] = DatasetInstance(video,
                                                                                                      annotations_path)