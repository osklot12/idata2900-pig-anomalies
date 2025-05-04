import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.utils.path_finder import PathFinder


class FeatureMapMaker:
    """Creates feature maps from filters."""

    @staticmethod
    def make_map(kernel: np.ndarray, img_path: str, output_path: str):
        """
        Creates a feature map for an image given a kernel (filter).

        Args:
            kernel (np.ndarray): the kernel to apply
            img_path (str): path to the image to make feature map for
            output_path (str): path to the output image
        """
        img = cv2.imread(str(PathFinder.get_abs_path(img_path)), cv2.IMREAD_GRAYSCALE)

        feature_map = cv2.filter2D(img, -1, kernel)
        print("Feature map min/max:", feature_map.min(), feature_map.max())
        feature_map = np.abs(feature_map * 2.0).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        feature_map = clahe.apply(feature_map)

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(img, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Feature Map (Sobel Y)")
        plt.imshow(feature_map, cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig("feature_map_example.png", dpi=300)
        plt.show()

    @staticmethod
    def _contrast_stretch(img):
        min_val = np.min(img)
        max_val = np.max(img)
        stretched = (img - min_val) * (255.0 / (max_val - min_val))
        return np.clip(stretched, 0, 255).astype(np.uint8)

    @staticmethod
    def _adjust_gamma(image, gamma=0.5):  # < 1 brightens
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255
                          for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)


