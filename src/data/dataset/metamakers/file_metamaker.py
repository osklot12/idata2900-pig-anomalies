from typing import Dict, List
from tqdm import tqdm

from src.data.dataset.metamakers.metamaker import Metamaker, K, L
from src.data.dataset.registries.suffix_file_registry import SuffixFileRegistry
from src.data.dataset.selectors.determ_string_selector import DetermStringSelector
from src.data.dataset.selectors.selector import Selector
from src.data.loading.loaders.factories.loader_factory import LoaderFactory


# key type for instance
# K = TypeVar("K")

# label class type
# L = TypeVar("L")


class FileMetamaker(Metamaker):
    """Generates metadata from annotation files."""

    def __init__(self, loader_factory: LoaderFactory):
        """
        Initializes a FetaMaker instance.

        Args:
            loader_factory (LoaderFactory): factory for creating data loaders
        """
        self._loader_factory = loader_factory

    def make_metadata(self) -> Dict[K, Dict[L, int]]:
        metadata: Dict[K, Dict[L, int]] = {}

        registry = self._loader_factory.create_file_registry()
        anno_registry = SuffixFileRegistry(source=registry, suffixes=("json",))
        annotation_loader = self._loader_factory.create_annotation_loader()

        files = anno_registry.get_file_paths()
        selector = DetermStringSelector(files)
        ids = self._get_annotations_ids(selector)

        for id_ in tqdm(ids, desc="Generating metadata"):
            annotations = annotation_loader.load_video_annotations(id_)

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