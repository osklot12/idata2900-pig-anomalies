from typing import Optional, Iterator

from src.data.dataclasses.dataset_instance import DatasetInstance
from src.data.dataset.manifests.manifest import Manifest
from src.data.dataset.providers.instance_provider import InstanceProvider
from src.data.dataset.selectors.selector import Selector


class ManifestInstanceProvider(InstanceProvider):
    """Provides dataset instances from a dataset manifest."""

    def __init__(self, manifest: Manifest, selector: Selector):
        """
        Initializes a ManifestInstanceProvider instance.

        Args:
            manifest (Manifest): the dataset manifest to get instances from
            selector (Selector): a string selector determines what instances to provide
        """
        self._manifest = manifest
        self._selector = selector

    def get(self) -> Optional[DatasetInstance]:
        instance = None

        id_ = self._selector.select()
        if id_ is not None:
            instance = self._manifest.get_instance(id_)
            if instance is None:
                raise RuntimeError(f"Manifest does not have an instance for id {id_}")

        if instance is None:
            print(f"[ManifestInstanceProvider] End of stream")
            
        return instance