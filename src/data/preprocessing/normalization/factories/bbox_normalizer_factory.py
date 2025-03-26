from abc import ABC, abstractmethod

from src.data.preprocessing.normalization.normalizers.bbox_normalizer import BBoxNormalizer


class BBoxNormalizerFactory(ABC):
    """An interface for BBoxNormalizationStrategy factories."""

    @abstractmethod
    def create_bbox_normalizer(self) -> BBoxNormalizer:
        """
        Creates a BBoxNormalizationStrategy instance.

        Returns:
            BBoxNormalizer: the BBoxNormalizationStrategy instance
        """
        raise NotImplementedError