from src.data.dataclasses.bbox import BBox
from src.models.converters.bbox_scaler import BBoxScaler


class StaticBBoxScaler(BBoxScaler):
    """Scales bounding boxes with a static scaling factors."""

    def __init__(self, x_scalar: float, y_scalar: float):
        """
        Initializes a StaticBBoxScaler instance.

        Args:
            x_scalar (float): the horizontal scaling factor
            y_scalar (float): the vertical scaling factor
        """
        self._x_scalar = x_scalar
        self._y_scalar = y_scalar

    def scale(self, bbox: BBox) -> BBox:
        return BBox(
            x=bbox.x * self._x_scalar,
            y=bbox.y * self._y_scalar,
            width=bbox.width * self._x_scalar,
            height=bbox.height * self._y_scalar
        )