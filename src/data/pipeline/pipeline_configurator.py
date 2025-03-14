from abc import ABC, abstractmethod

from src.data.streaming.streamers import AnnotationStreamer
from src.data.streaming.streamers import FrameStreamer
from src.data.streaming.aggregators.buffered_instance_aggregator import BufferedInstanceAggregator
from src.data.dataset.virtual_dataset import VirtualDataset


class PipelineConfigurator(ABC):
    """An interface for classes that configures pipelines, encapsulating configuration algorithms."""

    @abstractmethod
    def config_frame_loader(self) -> FrameStreamer:
        """Configures and returns a FrameLoader object."""
        raise NotImplementedError

    @abstractmethod
    def config_annotation_loader(self) -> AnnotationStreamer:
        """Configures and returns a AnnotationLoader object."""
        raise NotImplementedError

    @abstractmethod
    def config_instance_loader(self) -> BufferedInstanceAggregator:
        """Configures and returns a InstanceLoader object."""
        raise NotImplementedError

    @abstractmethod
    def config_virtual_dataset(self) -> VirtualDataset:
        """Configures and returns a VirtualDataset object."""
        raise NotImplementedError