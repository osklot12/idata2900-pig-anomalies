import pytest
import os
import json
import shutil
import tensorflow as tf
from src.data.data_manager import DataManager

# Test paths
TEST_FRAME_DIR = "../../data/mock_frame_extractor/mock_frames_manager"
TEST_ANNOTATION_PATH_VIDEO_1 = "../../data/mock_frame_extractor/mock_annotations_video_1.json"
TEST_ANNOTATION_PATH_VIDEO_2 = "../../data/mock_frame_extractor/mock_annotations_video_2.json"
TEST_OUTPUT_DIR = "../../delete_later"

@pytest.fixture
def setup_test_env():
    """Setup test directories and mock frame files."""
    if os.path.exists(TEST_OUTPUT_DIR):
        shutil.rmtree(TEST_OUTPUT_DIR)

    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    yield  # Let the test run

    #shutil.rmtree(TEST_OUTPUT_DIR)

@pytest.fixture
def mock_data_manager():
    """Create DataManager instance for testing."""
    return DataManager(output_dir=TEST_OUTPUT_DIR, num_augmented_versions=10)

@pytest.fixture
def mock_frame_data():
    """
    Loads JPG frames as in-memory tensors to simulate actual usage.
    :return: Dictionary of {frame_name: tensor}.
    """
    frame_data = {}
    for i in range(1, 6):  # 5 frames per video
        frame_path = os.path.join(TEST_FRAME_DIR, f"frame_{i}_video_1.jpg")
        frame_data[f"frame_{i}_video_1.jpg"] = tf.image.decode_jpeg(tf.io.read_file(frame_path), channels=3)

    return frame_data

@pytest.fixture
def mock_annotations():
    """Loads mock annotation data."""
    with open(TEST_ANNOTATION_PATH_VIDEO_1, "r") as f:
        annotation_data_1 = json.load(f)

    with open(TEST_ANNOTATION_PATH_VIDEO_2, "r") as f:
        annotation_data_2 = json.load(f)

    return {"video_1": annotation_data_1, "video_2": annotation_data_2}

def test_data_manager_frame_processing(setup_test_env, mock_data_manager, mock_frame_data, mock_annotations):
    """Test if DataManager correctly processes frames and maintains order."""

    # Set environment variable to force local annotation loading
    os.environ["TEST_MODE"] = "true"

    # Process video_1
    mock_data_manager.process_video_frames("video_1", mock_frame_data, mock_annotations["video_1"])

    # Verify that 10 versions of each video are saved correctly
    for version in range(10):
        version_dir = os.path.join(TEST_OUTPUT_DIR, f"video_1_version{version}")
        assert os.path.exists(version_dir), f"❌ Augmented version {version} of video_1 is missing"

        # Check for 5 frames per version
        frame_files = sorted(
            [f for f in os.listdir(version_dir) if f.endswith(".jpg")],
            key=lambda x: int(x.split("_")[1].split(".")[0])
        )
        assert len(frame_files) == 5, f"❌ Expected 5 frames in {version_dir}, found {len(frame_files)}"

        # Check for JSON annotation file
        json_path = os.path.join(version_dir, "annotations.json")
        assert os.path.exists(json_path), f"❌ Missing JSON file in {version_dir}"

        # Validate JSON structure
        with open(json_path, "r") as f:
            annotation_data = json.load(f)
            assert isinstance(annotation_data, dict), f"❌ JSON annotations not structured correctly in {version_dir}"

    # Clean up
    del os.environ["TEST_MODE"]
