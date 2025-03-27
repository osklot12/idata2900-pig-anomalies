from abc import ABC, abstractmethod

from src.data.frame_instance_provider import FrameInstanceProvider


class ServerContext(ABC):
    """An interface for server contexts."""

    @abstractmethod
    def get_frame_instance_provider(self) -> FrameInstanceProvider:
        """
        Returns instance of FrameInstanceProvider.

        Returns:
            FrameInstanceProvider: An instance of FrameInstanceProvider.
        """
        raise NotImplementedError