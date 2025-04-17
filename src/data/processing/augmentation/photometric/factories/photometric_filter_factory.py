from abc import ABC, abstractmethod

from src.data.processing.augmentation.photometric.photometric_filter import PhotometricFilter


class PhotometricFilterFactory(ABC):
    """Interface for factories of PhotometricFilter instances."""

    @abstractmethod
    def create_filter(self) -> PhotometricFilter:
        """
        Creates and returns a PhotometricFilter instance.

        Returns:
            PhotometricFilter: a PhotometricFilter instance
        """
        raise NotImplementedError