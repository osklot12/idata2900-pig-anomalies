from enum import Enum
from typing import Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.data.dataclasses.bbox import BBox
from src.data.dataclasses.annotated_frame import AnnotatedFrame


class AnnotatedFrameVisualizer:
    """Visualizer for AnnotatedFrame instances."""

    @staticmethod
    def visualize(instance: AnnotatedFrame) -> None:
        """
        Visualizes an AnnotatedFrame instance.

        Args:
            instance (AnnotatedFrame): the annotated frame to visualize
        """
        frame = np.copy(instance.frame)
        for annotation in instance.annotations:
            x_min, y_min, x_max, y_max = AnnotatedFrameVisualizer._get_absolute_coordinates(
                annotation.bbox, frame.shape[1], frame.shape[0]
            )
            AnnotatedFrameVisualizer._draw_bbox(frame, annotation.cls, x_min, y_min, x_max, y_max)

        AnnotatedFrameVisualizer._show_plot(
            frame=frame,
            frame_index=instance.index,
            source=instance.source.source_id,
            annotated=len(instance.annotations) > 0
        )

    @staticmethod
    def _get_absolute_coordinates(bbox: BBox, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        """Computes the absolute coordinates for a normalized bounding box based on original width and height."""
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
    def _draw_bbox(frame: np.ndarray, behavior: Enum, x_min: int, y_min: int, x_max: int, y_max: int) -> None:
        """Draws a bounding box."""
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(
            frame, behavior.name.replace("_", " "), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 1, cv2.LINE_AA
        )

    @staticmethod
    def _show_plot(frame: np.ndarray, frame_index: int, source: str, annotated: bool) -> None:
        """Displays the plot."""
        plt.figure(figsize=(6, 6))
        plt.imshow(frame)

        if annotated:
            title = f"{source} ({frame_index}) [Annotated]"
        else:
            title = f"{source} ({frame_index})"
        plt.title(title)

        plt.axis("off")
        plt.show()
