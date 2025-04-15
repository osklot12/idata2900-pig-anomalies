import numpy as np


class Transformator:
    """Computes transformation matrices."""

    @staticmethod
    def compute_shift_and_transform_matrix(t: np.ndarray, cx: float, cy: float) -> np.ndarray:
        """
        Computes a transformation matrix that shifts to the origin before transforming.

        Args:
            t (np.ndarray): the transformation to apply
            cx (float): the x-coordinate of the center
            cy (float): the y-coordinate of the center
        """
        return (
                Transformator.compute_translation_matrix(cx, cy) @
                t @
                Transformator.compute_translation_matrix(-cx, -cy)
        )

    @staticmethod
    def compute_reflection_matrix() -> np.ndarray:
        """
        Computes a reflection matrix.

        Returns:
            np.ndarray: the reflection matrix
        """
        return np.array([
            [-1, 0, 0],
            [0, 1, 0, ],
            [0, 0, 1]
        ], dtype=np.float32)

    @staticmethod
    def compute_translation_matrix(dx: float, dy: float) -> np.ndarray:
        """
        Computes the translation matrix for a shift by (dx, dy).

        Args:
            dx (float): the horizontal translation distance
            dy (float): the vertical translation distance

        Returns:
            np.ndarray: the translation matrix
        """
        return np.array([
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1]
        ], dtype=np.float32)

    @staticmethod
    def compute_dilation_matrix(factor: float) -> np.ndarray:
        """
        Computes the dilation matrix from given dilation factor.

        Args:
            factor (float): the scaling factor

        Returns:
            np.ndarray: the dilation matrix
        """
        return np.array([
            [factor, 0, 0],
            [0, factor, 0],
            [0, 0, 1]
        ], dtype=np.float32)

    @staticmethod
    def compute_rotation_matrix(degrees: float) -> np.ndarray:
        """
        Computes the rotation matrix from given degrees.

        Args:
            degrees (float): the rotation angle (counterclockwise)

        Returns:
            np.ndarray: the rotation matrix
        """
        radians = np.deg2rad(degrees)
        cos_theta = np.cos(radians)
        sin_theta = np.sin(radians)

        return np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ], dtype=np.float32)