from typing import List

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.preprocessing.augmentation.augmentors.annotations_augmentor import AnnotationsAugmentor
from src.data.preprocessing.preprocessor import Preprocessor
from src.data.processing.augmentor import Augmentor
from src.data.preprocessing.augmentation.plan.augmentation_plan_factory import AugmentationPlanFactory


class AnnotatedFrameAugmentor(Preprocessor[AnnotatedFrame]):
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

    def process(self, instance: AnnotatedFrame) -> List[AnnotatedFrame]:
        augmentations = []

        frame_shape = (instance.frame.shape[1], instance.frame.shape[0])
        frame_augmentor = Augmentor(self._plan_factory)
        annotations_aug = AnnotationsAugmentor(self._plan_factory, frame_shape)

        for _ in range(self._outputs):
            with self._plan_factory:
                augmentations.append(
                    AnnotatedFrame(
                        source=instance.source,
                        index=instance.index,
                        frame=frame_augmentor.process(instance.frame)[0],
                        annotations=annotations_aug.process(instance.annotations)[0]
                    )
                )

        return augmentations