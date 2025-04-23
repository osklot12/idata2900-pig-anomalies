import json
from typing import Dict, List
from tqdm import tqdm

from src.data.dataset.metamakers.metamaker import Metamaker, K, L
from src.data.dataset.registries.suffix_file_registry import SuffixFileRegistry
from src.data.dataset.selectors.determ_string_selector import DetermStringSelector
from src.data.dataset.selectors.selector import Selector
from src.data.loading.loaders.factories.loader_factory import LoaderFactory
from src.data.parsing.base_name_parser import BaseNameParser
from src.utils.path_finder import PathFinder


# key type for instance
# K = TypeVar("K")

# label class type
# L = TypeVar("L")


class FileMetamaker(Metamaker):
    """Generates metadata from annotation files."""

    def __init__(self, loader_factory: LoaderFactory, cache: bool = False, cache_dir: str = "cache/metadata.json"):
        """
        Initializes a FetaMaker instance.

        Args:
            loader_factory (LoaderFactory): factory for creating data loaders
            cache (bool): whether to cache the metadata
            cache_dir (str): directory to cache the metadata
        """
        self._loader_factory = loader_factory
        self._cache = cache
        self._cache_dir = cache_dir
        self._parser = BaseNameParser()

    def make_metadata(self) -> Dict[K, Dict[L, int]]:
        return self._get_cached_metadata() if self._cache else self._generate_metadata()

    def _get_cached_metadata(self) -> Dict[K, Dict[L, int]]:
        """Tries to fetch cached metadata, or creates it if it does not exist."""
        metadata: Dict[K, Dict[L, int]] = {}

        output_path = PathFinder.get_abs_path(self._cache_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not output_path.exists():
            metadata = self._generate_metadata()

            metadata_serializable = {
                str(k): {str(label): count for label, count in labels.items()}
                for k, labels in metadata.items()
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(metadata_serializable, f, indent=2)

        else:
            with open(output_path, "r", encoding="utf-8") as f:
                metadata: Dict[str, Dict[str, int]] = json.load(f)

        return metadata

    def _generate_metadata(self) -> Dict[K, Dict[L, int]]:
        """Generates metadata."""
        metadata: Dict[K, Dict[L, int]] = {}

        registry = self._loader_factory.create_file_registry()
        anno_registry = SuffixFileRegistry(source=registry, suffixes=("json",))
        annotation_loader = self._loader_factory.create_annotation_loader()

        files = anno_registry.get_file_paths()
        selector = DetermStringSelector(files)
        ids = self._get_annotations_ids(selector)

        for id_ in tqdm(ids, desc="Generating metadata"):
            annotations = annotation_loader.load_video_annotations(id_)

            id_ = self._parser.parse_string(id_)
            if id_ not in metadata:
                metadata[id_] = {}
            label_counts = metadata.get(id_)

            for ans in annotations:
                for a in ans.annotations:
                    label = a.cls
                    if label not in label_counts:
                        label_counts[label] = 0

                    label_counts[label] += 1

        return metadata

    @staticmethod
    def _get_annotations_ids(selector: Selector[str]) -> List[str]:
        """Returns the IDs for the annotations files."""
        ids = []

        id_ = selector.select()
        while id_:
            ids.append(id_)
            id_ = selector.select()

        return ids