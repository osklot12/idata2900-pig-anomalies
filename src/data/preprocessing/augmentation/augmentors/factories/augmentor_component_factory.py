from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.pipeline.component import Component
from src.data.pipeline.component_factory import ComponentFactory
from src.data.preprocessing.augmentation.augmentors.augmentor_component import AugmentorComponent
from src.data.preprocessing.augmentation.augmentors.factories.augmentor_factory import AugmentorFactory


class AugmentorComponentFactory(ComponentFactory[StreamedAnnotatedFrame]):
    """Factory for creating InstanceAugmentorComponent instances."""

    def __init__(self, augmentor_factory: AugmentorFactory[StreamedAnnotatedFrame]):
        """
        Initializes an InstanceAugmentorComponentFactory instance.

        Args:
            augmentor_factory (AugmentorFactory[StreamedAnnotatedFrame]): factory for creating augmentors
        """
        self._augmentor_factory = augmentor_factory

    def create_component(self) -> Component[StreamedAnnotatedFrame]:
        return AugmentorComponent(augmentor=self._augmentor_factory.create_augmentor())
