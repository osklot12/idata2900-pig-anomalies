from typing import TypeVar, Generic, List

from src.data.dataclasses.dataset_split_ratios import DatasetSplitRatios
from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.managed.factories.gcs_stream_factory import GCSStreamFactory
from src.data.dataset.streams.managed.factories.split_stream_factory import SplitStreamFactory
from src.data.dataset.streams.managed.managed_stream import ManagedStream
from src.data.pipeline.component_factory import ComponentFactory
from src.utils.gcs_credentials import GCSCredentials
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass

T = TypeVar("T")


class NorsvinStreamFactory(Generic[T], SplitStreamFactory[T]):
    """Factory for creating dataset split streams for Norsvin dataset."""

    def __init__(self, gcs_creds: GCSCredentials, split_ratios: DatasetSplitRatios,
                 preprocessor_factories: List[ComponentFactory[T]]):
        """
        Initializes a NorsvinStreamFactory instance.

        Args:
            gcs_creds (GCSCredentials): Google Cloud Storage credentials
            split_ratios (DatasetSplitRatios): the split ratios for the dataset
            preprocessor_factories (List[ComponentFactory]): factories for creating preprocessors, in the particular order
        """
        self._gcs_creds = gcs_creds
        self._split_ratios = split_ratios
        self._preprocessor_factories = preprocessor_factories

    def create_stream(self, split: DatasetSplit) -> ManagedStream[T]:
        return GCSStreamFactory(
            gcs_creds=self._gcs_creds,
            split_ratios=self._split_ratios,
            split=split,
            label_map=NorsvinBehaviorClass.get_label_map(),
            preprocessor_factories=self._preprocessor_factories
        ).create_stream()