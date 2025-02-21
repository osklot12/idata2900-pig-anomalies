import pytest
import tensorflow as tf
from src.data.shuffler import Shuffler

@pytest.fixture
def mock_data():
    """Creates mock dataset with frames from two different 'videos'."""
    frames_video_1 = [
        {"video_id": "video_1", "frame": tf.zeros((224, 224, 3), dtype=tf.uint8), "frame_idx": i}
        for i in range(1, 6)
    ]
    frames_video_2 = [
        {"video_id": "video_2", "frame": tf.zeros((224, 224, 3), dtype=tf.uint8), "frame_idx": i}
        for i in range(6, 11)
    ]

    annotations_video_1 = {
        "annotations": [{"frames": {str(i): {"bounding_box": {"x": i*10, "y": i*10, "w": 50, "h": 50}, "text": {"text": "pig"}}} for i in range(1, 6)}]
    }
    annotations_video_2 = {
        "annotations": [{"frames": {str(i): {"bounding_box": {"x": i*20, "y": i*20, "w": 100, "h": 100}, "text": {"text": "cow"}}} for i in range(6, 11)}]
    }

    return (frames_video_1, frames_video_2), {"video_1": annotations_video_1, "video_2": annotations_video_2}

def test_shuffler(mock_data):
    """Tests if Shuffler properly shuffles frames and generates a valid COCO annotation file."""
    (frames_video_1, frames_video_2), annotations = mock_data
    all_frames = frames_video_1 + frames_video_2

    shuffler = Shuffler(seed=42)
    shuffled_frames, merged_annotations = shuffler.shuffle(all_frames, annotations)

    assert len(shuffled_frames) == len(all_frames), "❌ Data length changed after shuffling"
    assert len(merged_annotations["images"]) == len(shuffled_frames), "❌ Image count mismatch in JSON"
    assert len(merged_annotations["annotations"]) == len(shuffled_frames), "❌ Annotation count mismatch in JSON"

    print("✅ Shuffling correctly preserves frame-annotation links and outputs merged JSON.")
