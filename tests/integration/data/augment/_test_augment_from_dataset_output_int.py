import numpy as np

from src.data.dataset.virtual.virtual_dataset import VirtualDataset
from src.data.preprocessing.augmentation.combined_augmentor import CombinedAugmentation


def create_dummy_frame(shape=(640, 640, 3)):
    return np.random.randint(0, 256, size=shape, dtype=np.uint8)


def create_dummy_annotations():
    return [("pig", 100.0, 150.0, 50.0, 60.0)]


def test_combined_augmentation_with_virtual_dataset():
    dummy_source = "video_01"

    # Ensure 100% train ratio so "video_01" lands in the train set
    dataset = VirtualDataset(
        dataset_ids=[dummy_source],
        max_sources=1,
        max_frames_per_source=10,
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0
    )

    augmentor = CombinedAugmentation(num_versions=2)

    frame = create_dummy_frame()
    annotations = create_dummy_annotations()

    augmented_versions = augmentor.augment(frame, annotations)

    for i, (aug_frame, aug_annotations) in enumerate(augmented_versions):
        assert isinstance(aug_frame, np.ndarray)
        assert isinstance(aug_annotations, list)
        assert all(isinstance(bbox, tuple) and len(bbox) == 5 for bbox in aug_annotations)
        dataset.feed(dummy_source, i, aug_frame, aug_annotations, end_of_stream=False)

    batch = dataset.get_shuffled_batch(DatasetSplit.TRAIN, batch_size=2)

    assert len(batch) == 2
    for frame_data, bboxes in batch:
        assert isinstance(frame_data, np.ndarray)
        assert isinstance(bboxes, list)
        assert all(len(bbox) == 5 and isinstance(bbox[0], str) for bbox in bboxes)
