from typing import Tuple


from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.dataset.virtual.virtual_dataset import VirtualDataset, O


class FrameDataset(VirtualDataset[StreamedAnnotatedFrame, AnnotatedFrame]):
    """A virtual dataset storing single annotated frames."""

    def _get_instance_id(self, food: StreamedAnnotatedFrame) -> str:
        return f"{food.get_id()}_f{str(food.index)}"

    def _frame_in_instance(self, instance: O) -> int:
        return 1