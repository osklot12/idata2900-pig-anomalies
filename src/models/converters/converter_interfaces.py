from abc import ABC
from typing import List, Dict

import torch

class BatchToTupleConverter(ABC):
    """
    An interface for converting batches of data into tensors and corresponding bounding boxes.
    """
    def convert_to_tuple_of_tensors(self, batch) -> tuple[list[torch.Tensor], list[Dict]]:
        """
        Converts a batch of data into tensors and corresponding bounding boxes.
        """
        raise NotImplementedError

class TargetToTensorConverter(ABC):
    """
    An interface for converting targets and images into tensors.
    """
    def convert_to_tensors(self, targets: List[Dict], images: List[torch.Tensor]) -> torch.Tensor:
        """
        Converts targets and images into tensors.
        """
        raise NotImplementedError

class PredictionsToListConverter(ABC):
    """
    An interface for converting predictions from YOLO models into List[Dict].
    """
    def convert_preds_to_list(self, preds) -> List[Dict]:
        """
        Converts YOLO predictions into List[Dict].
        """
        raise NotImplementedError
