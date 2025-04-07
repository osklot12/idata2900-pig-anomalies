from typing import Tuple

import cv2

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataclasses.bbox import BBox


class AnnotatedFrameVisualizer:
    """Visualizer for AnnotatedFrame instances."""

    @staticmethod
    def visualize(instance: AnnotatedFrame) -> None:
        """
        Visualizes an AnnotatedFrame instance.

        Args:
            instance (AnnotatedFrame): the annotated frame to visualize
        """
        for annotation in instance.annotations:
            AnnotatedFrameVisualizer._draw_bbox(
                annotation.cls, *AnnotatedFrameVisualizer._get_absolute_coordinates(
                    annotation.bbox, instance.
                )
            )

    @staticmethod
    def _get_absolute_coordinates(bbox: BBox, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        x = bbox.x
        y = bbox.y
        w = bbox.width
        h = bbox.height

        x_min = int(x * frame_width)
        x_max = int((x + w) * frame_width)
        y_min = int(y * frame_height)
        y_max = int((y + h) * frame_height)

        return x_min, y_min, x_max, y_max

    @staticmethod
    def _draw_bbox(behavior, frame, x_max, x_min, y_max, y_min):
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(
            frame, behavior.name.replace("_", " "), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 1, cv2.LINE_AA
        )

    @staticmethod
    def _show_plot(frame, frame_index):
        plt.figure(figsize=(6, 6))
        plt.imshow(frame)
        plt.title(f"Frame {frame_index}")
        plt.axis("off")
        plt.show()