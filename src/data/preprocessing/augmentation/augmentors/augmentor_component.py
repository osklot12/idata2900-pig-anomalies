from typing import Optional

from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.pipeline.component import Component
from src.data.pipeline.consumer import Consumer
from src.data.preprocessing.augmentation.augmentors.augmentor import Augmentor
from src.data.structures.atomic_var import AtomicVar

class AugmentorComponent(Component[StreamedAnnotatedFrame]):
    """Pipeline component adapter for instance augmentation."""

    def __init__(self, augmentor: Augmentor[StreamedAnnotatedFrame], consumer: Optional[Consumer[StreamedAnnotatedFrame]] = None):
        """
        Initializes an InstanceAugmentorComponent instance.

        Args:
            augmentor (Augmentor[StreamedAnnotatedFrame]): augmentor to use for augmentation
            consumer (Optional[Consumer[StreamedAnnotatedFrame]]): optional consumer for receiving augmented data
        """
        self._augmentor = augmentor
        self._consumer = AtomicVar[Consumer[StreamedAnnotatedFrame]](consumer)

    def consume(self, data: Optional[StreamedAnnotatedFrame]) -> bool:
        success = False

        consumer = self._consumer.get()
        if consumer is not None:
            if data is not None:
                data = self._augmentor.augment(data)
            success = consumer.consume(data)

        return success


    def connect(self, consumer: Consumer[StreamedAnnotatedFrame]) -> None:
        self._consumer.set(consumer)