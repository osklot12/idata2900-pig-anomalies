import numpy as np
import pytest
from src.data.augment.image_resizer import ImageResizer


@pytest.mark.parametrize("target_size", [(128, 128), (256, 256)])
def test_resize(target_size):
    """Test if the image resizes correctly and annotations scale accordingly."""
    resizer = ImageResizer(target_size=target_size)
    sample_image = np.ones((512, 512, 3), dtype=np.uint8) * 255  # White image
    sample_annotations = [(1, 50, 50, 100, 100)]  # A bounding box

    resized_image, resized_annotations = resizer.process(sample_image, sample_annotations)

    assert resized_image.shape[:2] == target_size
    scale_x = target_size[0] / 512
    scale_y = target_size[1] / 512
    expected_annotations = [(1, 50 * scale_x, 50 * scale_y, 100 * scale_x, 100 * scale_y)]

    assert np.allclose(resized_annotations, expected_annotations), "Annotation scaling failed!"
