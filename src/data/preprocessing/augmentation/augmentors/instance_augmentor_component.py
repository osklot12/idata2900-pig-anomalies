from typing import Optional

from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.pipeline.consumer import Consumer
from src.data.pipeline.producer import Producer
from src.data.preprocessing.augmentation.augmentors.augmentor import Augmentor
from src.data.preprocessing.augmentation.plan.augmentation_plan_factory import AugmentationPlanFactory
from src.data.structures.atomic_var import AtomicVar

class InstanceAugmentorComponent(Consumer[StreamedAnnotatedFrame], Producer[StreamedAnnotatedFrame]):
    """Pipeline component adapter for instance augmentation."""

    def __init__(self, augmentor: Augmentor[StreamedAnnotatedFrame], plan_factory: AugmentationPlanFactory,
                 consumer: Optional[Consumer[StreamedAnnotatedFrame]] = None):
        """
        Initializes an InstanceAugmentorComponent instance.

        Args:
            augmentor (Augmentor[StreamedAnnotatedFrame]): augmentor to use for augmentation
            plan_factory (AugmentationPlanFactory): factory for creating augmentation plans
            consumer (Optional[Consumer[StreamedAnnotatedFrame]]): optional consumer for receiving augmented data
        """
        self._augmentor = augmentor
        self._plan_factory = plan_factory
        self._consumer = AtomicVar[Consumer[StreamedAnnotatedFrame]](consumer)

    def consume(self, data: Optional[StreamedAnnotatedFrame]) -> bool:
        success = False

        consumer = self._consumer.get()
        if consumer is not None:
            with self._plan_factory:
                augmented = self._augmentor.augment(data)
                success = consumer.consume(augmented)

        return success


    def connect(self, consumer: Consumer[T]) -> None:
        self._consumer.set(consumer)