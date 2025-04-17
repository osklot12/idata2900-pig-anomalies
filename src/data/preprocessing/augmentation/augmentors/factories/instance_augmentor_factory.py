from typing import Optional, List

from src.data.preprocessing.augmentation.augmentors.augmentor import Augmentor
from src.data.preprocessing.augmentation.augmentors.factories.augmentor_factory import AugmentorFactory, T
from src.data.processing.augmentor import Augmentor
from src.data.preprocessing.augmentation.augmentors.photometric.factories.photometric_filter_factory import \
    PhotometricFilterFactory
from src.data.preprocessing.augmentation.plan.augmentation_plan_factory import AugmentationPlanFactory


class InstanceAugmentorFactory(AugmentorFactory):
    """Factory for creating InstanceAugmentor instances."""

    def __init__(self, filter_factories: Optional[List[PhotometricFilterFactory]] = None):
        """
        Initializes an InstanceAugmentorFactory instance.

        Args:
            filter_factories (Optional[List[PhotometricFilterFactory]]): factories for creating photometric filters
        """
        self.filter_factories = filter_factories if filter_factories is not None else []

    def create_augmentor(self) -> Augmentor[T]:
        return Augmentor(
            plan_factory=AugmentationPlanFactory(),
            filters=[factory.create_filter() for factory in self.filter_factories]
        )