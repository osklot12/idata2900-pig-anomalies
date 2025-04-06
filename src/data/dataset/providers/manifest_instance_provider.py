from typing import Optional, Iterator

from src.data.dataclasses.dataset_instance import DatasetInstance
from src.data.dataset.manifests.manifest import Manifest
from src.data.dataset.providers.instance_provider import InstanceProvider
from src.data.dataset.selectors.string_selector import StringSelector


class ManifestInstanceProvider(InstanceProvider):
    """Provides dataset instances from a dataset manifest."""

    def __init__(self, manifest: Manifest, selector: StringSelector):
        """
        Initializes a ManifestInstanceProvider instance.

        Args:
            manifest (Manifest): the dataset manifest to get instances from
            selector (StringSelector): a string selector determines what instances to provide
        """
        self._manifest = manifest
        self._selector = selector

    def next(self) -> Optional[DatasetInstance]:
        instance = None

        id_ = self._selector.next()
        if id_ is not None:
            instance = self._manifest.get_instance(id_)
            if instance is None:
                raise RuntimeError(f"Manifest does not have an instance for id {id_}")

        return instance