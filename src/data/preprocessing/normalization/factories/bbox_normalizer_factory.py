from abc import ABC, abstractmethod

from src.data.preprocessing.normalization.normalizers.bbox_normalization_strategy import BBoxNormalizationStrategy


class BBoxNormalizerFactory(ABC):
    """An interface for BBoxNormalizationStrategy factories."""

    @abstractmethod
    def create_bbox_normalizer(self) -> BBoxNormalizationStrategy:
        """
        Creates a BBoxNormalizationStrategy instance.

        Returns:
            BBoxNormalizationStrategy: the BBoxNormalizationStrategy instance
        """
        raise NotImplementedError