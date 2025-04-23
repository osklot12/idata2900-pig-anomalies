import json
from pathlib import Path
from typing import List, Callable, Dict, TypeVar

from src.data.dataset.metamakers.file_metamaker import FileMetamaker
from src.data.dataset.registries.file_registry import FileRegistry
from src.data.loading.loaders.factories.loader_factory import LoaderFactory
from src.utils.path_finder import PathFinder

# label class type
L = TypeVar("L")


class SmartRegistry(FileRegistry):
    """Filters files based on metadata."""

    def __init__(self, loader_factory: LoaderFactory, filter_func: Callable[[Dict[L, int]], bool],
                 metadata_dir: str = "dataset_metadata/metadata.json"):
        """
        Initializes a SmartRegistry instance.

        Args:
            loader_factory (LoaderFactory): factory for creating loaders
            filter_func (Callable[[Dict[L, int]], bool]): function for filtering entries based on their metadata
        """
        self._loader_factory = loader_factory
        self._filter_func = filter_func
        self._metadata_dir = metadata_dir

    def get_file_paths(self) -> List[str]:
        output_path = PathFinder.get_abs_path(self._metadata_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not output_path.exists():
            self._generate_metadata(output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            metadata: Dict[str, Dict[str, int]] = json.load(f)

        selected_paths = [
            path for path, label_counts in metadata.items()
            if self._filter_func(label_counts)
        ]

        return selected_paths

    def _generate_metadata(self, path: Path) -> None:
        """Generates metadata for the dataset and saves it to the given path."""
        maker = FileMetamaker(self._loader_factory)
        metadata = maker.make_metadata()

        metadata_serializable = {
            str(k): {str(label): count for label, count in labels.items()}
            for k, labels in metadata.items()
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata_serializable, f, indent=2)
