from typing import Dict

from src.data.dataset.metamakers.metamaker import MetaMaker, K, L
from src.data.dataset.selectors.determ_string_selector import DetermStringSelector
from src.data.loading.loaders.factories.loader_factory import LoaderFactory


# key type for instance
# K = TypeVar("K")

# label class type
# L = TypeVar("L")


class FetaMaker(MetaMaker):
    """Generates metadata from annotation files."""

    def __init__(self, loader_factory: LoaderFactory):
        """
        Initializes a FetaMaker instance.

        Args:
            loader_factory (LoaderFactory): factory for creating data loaders
        """
        self._loader_factory = loader_factory


    def make_metadata(self) -> Dict[K, Dict[L, int]]:
        registry = self._loader_factory.create_file_registry()
        annotation_loader = self._loader_factory.create_annotation_loader()

        files = registry.get_file_paths()
        selector = DetermStringSelector(files)
        selected = selector.select()

