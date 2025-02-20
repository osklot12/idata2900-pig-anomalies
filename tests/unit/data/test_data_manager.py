import pytest
import json
import tensorflow as tf
from src.data.data_manager import DataManager

@pytest.fixture
def mock_data_manager():
    """Create DataManager instance for testing."""
    return DataManager(output_dir=None, num_augmented_versions=10)  # output_dir is not used anymore

@pytest.fixture
def mock_frame_data():
    """
    Generates mock image tensors to simulate actual JPG frames.
    :return: Dictionary of {frame_name: tensor}.
    """
    frame_data = {}
    for i in range(1, 6):  # 5 frames per video
        image_tensor = tf.random.uniform((224, 224, 3), dtype=tf.float32)  # Mock a valid image tensor
        frame_data[f"frame_{i}_video_1.jpg"] = image_tensor
    return frame_data

@pytest.fixture
def mock_annotations():
    """Returns mock annotation data."""
    return {
        "video_1": {
            "annotations": [{
                "frames": {
                    "1": {"label": "mock"},
                    "2": {"label": "mock"},
                    "3": {"label": "mock"},
                    "4": {"label": "mock"},
                    "5": {"label": "mock"}
                }
            }]
        }
    }

def test_data_manager_memory_storage(mock_data_manager, mock_frame_data, mock_annotations):
    """Test if DataManager correctly processes frames and stores them in memory."""

    # Process video_1
    mock_data_manager.process_video_frames("video_1", mock_frame_data, mock_annotations["video_1"])

    # Verify that 10 versions of each frame exist in memory
    assert len(mock_data_manager.memory_store) == 50, f"❌ Expected 50 frames in memory, found {len(mock_data_manager.memory_store)}"

    # Validate stored data
    for stored_entry in mock_data_manager.memory_store:
        assert "frame" in stored_entry, "❌ Missing frame in memory entry"
        assert "annotation" in stored_entry, "❌ Missing annotation in memory entry"

        # Ensure frame is a valid tensor
        assert isinstance(stored_entry["frame"], tf.Tensor), "❌ Frame is not a TensorFlow tensor"

        # Ensure annotation is valid JSON
        try:
            json.loads(stored_entry["annotation"])
        except json.JSONDecodeError:
            pytest.fail("❌ Annotation is not valid JSON")
