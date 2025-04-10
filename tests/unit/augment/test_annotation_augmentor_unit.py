from src.data.preprocessing.augmentation._annotation_augmentor import AnnotationAugmentor


def test_annotation_augmentation():
    """Test flipping and rotation operations on annotations."""
    augmentor = AnnotationAugmentor()
    sample_annotations = [(1, 50, 50, 100, 100)]
    image_shape = (224, 224)

    # Test flipping
    _, _, flipped_annotations = augmentor.augment("test_source", 1, sample_annotations, image_shape, flip=True)
    assert flipped_annotations[0][1] == image_shape[1] - (50 + 100), "Flipping annotations failed!"

    # Test rotation
    _, _, rotated_annotations = augmentor.augment("test_source", 1, sample_annotations, image_shape, rotation=15)
    assert rotated_annotations[0], "Rotation transformation failed!"
