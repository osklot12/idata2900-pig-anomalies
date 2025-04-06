from abc import ABC, abstractmethod
from typing import List, Optional

from src.data.dataclasses.dataset_instance import DatasetInstance


class Manifest(ABC):
    """Interface for dataset manifests."""

    @abstractmethod
    def list_all_ids(self) -> List[str]:
        """
        Returns a list of all instance IDs.

        Returns:
            List[str]: list of instance IDs
        """
        raise NotImplementedError

    @abstractmethod
    def get_instance(self, instance_id: str) -> Optional[DatasetInstance]:
        """
        Returns a dataset instance corresponding to the given instance ID.

        Args:
            instance_id (str): the ID of the instance

        Returns:
            DatasetInstance: the dataset instance, or None if no such ID exists
        """
        raise NotImplementedError
