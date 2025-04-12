from typing import List

from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.preprocessing.augmentation.augmentors.annotations_augmentor import AnnotationsAugmentor
from src.data.preprocessing.preprocessor import Preprocessor
from src.data.preprocessing.augmentation.augmentors.instance_augmentor import InstanceAugmentor
from src.data.preprocessing.augmentation.plan.augmentation_plan_factory import AugmentationPlanFactory


class AnnotatedFrameAugmentor(Preprocessor[StreamedAnnotatedFrame]):
    """Augments annotated frames."""

    def __init__(self, plan_factory: AugmentationPlanFactory, outputs: int = 1):
        """
        Initializes an AnnotatedFrameAugmentor instance.

        Args:
            plan_factory (AugmentationPlanFactory): factory for creating augmentation plans
            outputs (int): number of augmentations to create, defaults to 1
        """
        if outputs < 1:
            raise ValueError("Number of outputs must be greater than 0")

        self._plan_factory = plan_factory
        self._outputs = outputs

    def process(self, instance: StreamedAnnotatedFrame) -> List[StreamedAnnotatedFrame]:
        augmentations = []

        frame_shape = (instance.frame.shape[1], instance.frame.shape[0])
        frame_augmentor = InstanceAugmentor(self._plan_factory)
        annotations_aug = AnnotationsAugmentor(self._plan_factory, frame_shape)

        for _ in range(self._outputs):
            with self._plan_factory:
                augmentations.append(
                    StreamedAnnotatedFrame(
                        source=instance.source,
                        index=instance.index,
                        frame=frame_augmentor.process(instance.frame)[0],
                        annotations=annotations_aug.process(instance.annotations)[0]
                    )
                )

        return augmentations