from typing import Tuple

from src.data.dataclasses.bbox import BBox
from src.models.converters.bbox_converter import BBoxConverter


class BBoxToCorners(BBoxConverter):
    """Converts bounding boxes to (x1, y1), (x2, y2) format."""

    def convert(self, bbox: BBox) -> Tuple[float, float, float, float]:
        return bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height