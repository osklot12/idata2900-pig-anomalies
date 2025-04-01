from typing import Tuple


from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.dataset.virtual.virtual_dataset import VirtualDataset, O, I


class FrameDataset(VirtualDataset[StreamedAnnotatedFrame, AnnotatedFrame]):
    """A virtual dataset storing single annotated frames."""

    def _get_identified_instance(self, food: StreamedAnnotatedFrame) -> Tuple[str, AnnotatedFrame]:
        return food.get_id() + str(food.index), AnnotatedFrame(frame=food.frame, annotations=food.annotations)

    def _frame_in_instance(self, instance: O) -> int:
        return 1