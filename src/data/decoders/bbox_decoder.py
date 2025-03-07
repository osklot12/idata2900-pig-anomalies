from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Any

class BBoxDecoder(ABC):
    """Abstract base class for bounding box decoders."""

    def __init__(self, json_data: Dict[str, Any]):
        """
        Initializes the decoder with annotation data.
s
        Args:
            json_data (Dict[str, Any]): The annotation data.
        """
        self.json_data = json_data

    @abstractmethod
    def get_annotations(self) -> Dict[int, List[Tuple[str, float, float, float, float]]]:
        """Decodes annotations into a structured format."""
        raise NotImplementedError

    @abstractmethod
    def get_frame_count(self) -> int:
        """Returns the frame count from the annotation data."""
        raise NotImplementedError

    @abstractmethod
    def get_frame_dimensions(self) -> Tuple[int, int]:
        """Returns the width and height of the original video frames."""
        raise NotImplementedError