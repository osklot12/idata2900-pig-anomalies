from typing import Tuple


from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataset.virtual.virtual_dataset import VirtualDataset, O


class FrameDataset(VirtualDataset[AnnotatedFrame, AnnotatedFrame]):
    """A virtual dataset storing single annotated frames."""

    def _get_instance_id(self, food: AnnotatedFrame) -> str:
        return f"{food.get_id()}_f{str(food.index)}"

    def _frame_in_instance(self, instance: O) -> int:
        return 1