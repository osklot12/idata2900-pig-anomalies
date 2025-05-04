import cv2
import numpy as np

from src.utils.visualization.feature_map_maker import FeatureMapMaker


def main():
    kernel = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float32)

    FeatureMapMaker.make_map(kernel, "assets/pig.jpg", "assets")


if __name__ == "__main__":
    main()