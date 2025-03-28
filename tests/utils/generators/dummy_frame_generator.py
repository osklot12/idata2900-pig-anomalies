import numpy as np


class DummyFrameGenerator:
    """A generator of dummy frames."""

    @staticmethod
    def generate(width: int, height: int) -> np.ndarray:
        """
        Generates a dummy frame.

        Args:
            width (int): the width of the generated frame
            height (int): the height of the generated frame

        Returns:
            np.ndarray: the generated frame
        """
        return np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)