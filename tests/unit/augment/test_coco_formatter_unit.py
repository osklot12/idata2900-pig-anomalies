import pytest
import json
import tensorflow as tf
from src.data.preprocessing.augmentation.coco_formatter import COCOFormatter
from src.utils.path_finder import PathFinder


@pytest.fixture
def real_annotation_data():
    """Loads the real annotation file into memory."""
    annotation_path = PathFinder.get_abs_path("tests/data/annotations/annotation_darwin.json")
    with open(annotation_path, "r") as f:
        return json.load(f)  # Loads annotations into memory


@pytest.fixture
def mock_frame_data(real_annotation_data):
    """Creates mock frame data using real annotations."""
    frame_data = []

    for annotation in real_annotation_data["annotations"]:
        for frame_id, frame_info in annotation["frames"].items():
            if "bounding_box" in frame_info:
                frame_data.append({
                    "frame": tf.zeros((224, 224, 3), dtype=tf.uint8),  # Placeholder image tensor
                    "annotation": {frame_id: frame_info}  # Store frame-specific annotation
                })

    return frame_data


def test_coco_formatter(mock_frame_data):
    """Tests if COCOFormatter correctly converts a real annotation JSON file in memory."""
    formatter = COCOFormatter()
    coco_data = formatter.process(mock_frame_data)  # Uses in-memory annotations

    assert "images" in coco_data, "❌ Missing images in COCO JSON"
    assert "annotations" in coco_data, "❌ Missing annotations in COCO JSON"
    assert "categories" in coco_data, "❌ Missing categories in COCO JSON"
    assert len(coco_data["images"]) > 0, "❌ Expected image entries"
    assert len(coco_data["annotations"]) > 0, "❌ Expected annotation entries"
    assert len(coco_data["categories"]) > 0, "❌ Expected category entries"
