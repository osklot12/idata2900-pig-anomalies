import json
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List

from src.data.dataset.selectors.determ_string_selector import DetermStringSelector
from src.data.dataset.selectors.selector import Selector
from src.data.dataset.registries.suffix_file_registry import SuffixFileRegistry
from src.data.loading.loaders.factories.loader_factory import LoaderFactory
from src.data.parsing.base_name_parser import BaseNameParser

ANNOTATION_FILE_SUFFIXES = ["json"]
FILE_EXCEPTIONS = ["metadata"]


class FrameCounter:
    def __init__(self, loader_factory: LoaderFactory):
        self._loader_factory = loader_factory
        self._parser = BaseNameParser()

    def count_total_frames(self) -> int:
        """Sums the frame_count field across all annotation files."""
        total_frames = 0
        for raw in self._iterate_annotation_jsons():
            try:
                slots = raw["item"]["slots"]
                if isinstance(slots, list) and len(slots) > 0:
                    total_frames += slots[0].get("frame_count", 0)
            except (KeyError, TypeError):
                continue
        return total_frames

    def count_frames_per_class(self) -> Dict[str, int]:
        """
        Computes number of frames each class appears in.
        One frame may count toward multiple classes.
        """
        class_frame_map: Dict[str, set] = defaultdict(set)

        for raw in self._iterate_annotation_jsons():
            try:
                annotations = raw.get("annotations", [])
                for ann in annotations:
                    label = ann["name"]
                    frame_ids = ann.get("frames", {}).keys()
                    for fid in frame_ids:
                        class_frame_map[label].add(fid)
            except (KeyError, TypeError):
                continue

        return {cls: len(frames) for cls, frames in class_frame_map.items()}

    def _iterate_annotation_jsons(self) -> List[dict]:
        """Yields raw JSON data for each annotation file."""
        registry = self._loader_factory.create_file_registry()
        anno_registry = SuffixFileRegistry(source=registry, suffixes=tuple(ANNOTATION_FILE_SUFFIXES))
        annotation_loader = self._loader_factory.create_annotation_loader()

        files = anno_registry.get_file_paths()
        selector = DetermStringSelector(files)
        ids = self._get_annotation_ids(selector)

        valid_ids = [
            id_ for id_ in ids
            if self._parser.parse_string(id_) not in FILE_EXCEPTIONS
        ]

        for id_ in tqdm(valid_ids, desc="Loading annotations"):
            try:
                yield annotation_loader.load_raw_annotation(id_)
            except Exception as e:
                print(f"Failed to load {id_}: {e}")

    @staticmethod
    def _get_annotation_ids(selector: Selector[str]) -> List[str]:
        ids = []
        id_ = selector.select()
        while id_:
            ids.append(id_)
            id_ = selector.select()
        return ids
