from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.pipeline.component import Component
from src.data.pipeline.component_factory import ComponentFactory
from src.data.preprocessing.augmentation.augmentors.augmentor_component import AugmentorComponent
from src.data.preprocessing.augmentation.augmentors.factories.augmentor_factory import AugmentorFactory


class AugmentorComponentFactory(ComponentFactory[AnnotatedFrame, AnnotatedFrame]):
    """Factory for creating InstanceAugmentorComponent instances."""

    def __init__(self, augmentor_factory: AugmentorFactory[AnnotatedFrame]):
        """
        Initializes an InstanceAugmentorComponentFactory instance.

        Args:
            augmentor_factory (AugmentorFactory[AnnotatedFrame]): factory for creating augmentors
        """
        self._augmentor_factory = augmentor_factory

    def create_component(self) -> Component[AnnotatedFrame, AnnotatedFrame]:
        return AugmentorComponent(augmentor=self._augmentor_factory.create_augmentor())
