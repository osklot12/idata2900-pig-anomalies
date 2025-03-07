import cv2
import numpy as np

class ImageResizer:
    """
    Resizes images and scales annotations accordingly.
    """
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def process(self, image: np.ndarray, annotations: list):
        """
        Resizes the image and scales the annotations.

        :param image: Raw RGB image as a NumPy array.
        :param annotations: List of tuples (class, x, y, w, h).
        :return: Resized image and adjusted annotations.
        """
        original_h, original_w = image.shape[:2]
        resized_image = cv2.resize(image, self.target_size)

        scale_x = self.target_size[0] / original_w
        scale_y = self.target_size[1] / original_h

        resized_annotations = [
            (cls, x * scale_x, y * scale_y, w * scale_x, h * scale_y)
            for cls, x, y, w, h in annotations
        ]

        return resized_image, resized_annotations
