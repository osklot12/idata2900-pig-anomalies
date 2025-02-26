import numpy as np
from src.data.augment.combined_augmentor import CombinedAugmentation

def test_combined_augmentation():
    """Test full augmentation pipeline on images and annotations."""
    pipeline = CombinedAugmentation()
    sample_image = np.ones((224, 224, 3), dtype=np.uint8) * 255
    sample_annotations = [(1, 50, 50, 100, 100)]

    # Apply augmentation
    aug_image, aug_annotations = pipeline.augment(sample_image, sample_annotations)

    # Validate the image hasn't changed dimensions
    assert aug_image.shape == sample_image.shape, "Augmentation altered image dimensions!"

    # Validate annotations remain the correct format
    assert isinstance(aug_annotations, list), "Annotations should remain a list!"
    assert all(len(ann) == 5 for ann in aug_annotations), "Annotations should contain (class, x, y, w, h)!"
