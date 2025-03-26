import numpy as np
import pytest
from src.data.augment.combined_augmentor import CombinedAugmentation


@pytest.mark.parametrize("num_versions", [1, 3, 5])
def test_combined_augmentation(num_versions):
    """Test full augmentation pipeline with multiple augmentation versions."""

    augmentor = CombinedAugmentation(rotation_range=[-10, 10], num_versions=num_versions)

    sample_image = np.ones((224, 224, 3), dtype=np.uint8) * 255  # White image
    sample_annotations = [(1, 50, 50, 100, 100)]  # Example bounding box

    augmented_data = augmentor.augment(sample_image, sample_annotations)

    # Ensure correct number of augmented outputs
    assert len(augmented_data) == num_versions, f"Expected {num_versions} augmentations, got {len(augmented_data)}!"

    for aug_image, aug_annotations in augmented_data:
        assert aug_image.shape == sample_image.shape, "Augmentation altered image dimensions!"
        assert isinstance(aug_annotations, list), "Annotations should remain a list!"
        assert all(len(ann) == 5 for ann in aug_annotations), "Annotations should contain (class, x, y, w, h)!"


@pytest.mark.parametrize("rotation_range", [[-20, 20], [-5, 5], None])
def test_combined_augmentation_rotation(rotation_range):
    """Test augmentation rotation behavior with different rotation ranges."""

    augmentor = CombinedAugmentation(rotation_range=rotation_range, num_versions=2)

    sample_image = np.ones((224, 224, 3), dtype=np.uint8) * 255
    sample_annotations = [(1, 50, 50, 100, 100)]

    augmented_data = augmentor.augment(sample_image, sample_annotations)

    assert len(augmented_data) == 2, "Expected 2 augmentations!"

    for aug_image, aug_annotations in augmented_data:
        assert aug_image.shape == sample_image.shape, "Augmentation altered image dimensions!"
        assert isinstance(aug_annotations, list), "Annotations should remain a list!"
        assert all(len(ann) == 5 for ann in aug_annotations), "Annotations should contain (class, x, y, w, h)!"


def test_combined_augmentation_defaults():
    """Test augmentation pipeline defaults (should default to 1 augmentation)."""

    augmentor = CombinedAugmentation()  # Defaults to 1 augmentation

    sample_image = np.ones((224, 224, 3), dtype=np.uint8) * 255
    sample_annotations = [(1, 50, 50, 100, 100)]

    augmented_data = augmentor.augment(sample_image, sample_annotations)

    assert len(augmented_data) == 1, "Default augmentation should return exactly 1 version!"
