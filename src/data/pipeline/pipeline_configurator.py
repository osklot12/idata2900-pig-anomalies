from abc import ABC, abstractmethod

from src.data.streamers.annotation_streamer import AnnotationStreamer
from src.data.streamers.frame_streamer import FrameStreamer
from src.data.loading.instance_loader import InstanceLoader
from src.data.virtual_dataset import VirtualDataset


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
    def config_instance_loader(self) -> InstanceLoader:
        """Configures and returns a InstanceLoader object."""
        raise NotImplementedError

    @abstractmethod
    def config_virtual_dataset(self) -> VirtualDataset:
        """Configures and returns a VirtualDataset object."""
        raise NotImplementedError