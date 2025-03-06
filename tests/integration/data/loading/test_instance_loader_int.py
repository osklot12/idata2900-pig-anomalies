from threading import Event
from unittest.mock import Mock

import pytest

from src.data.loading.annotation_loader import AnnotationLoader
from src.data.loading.feed_status import FeedStatus
from src.data.loading.instance_loader import InstanceLoader
from src.data.loading.frame_loader import FrameLoader
from tests.utils.dummies.dummy_gcp_data_loader import DummyGCPDataLoader


@pytest.fixture
def dummy_data_loader():
    """Creates a dummy GCP data loader instance."""
    return DummyGCPDataLoader(bucket_name="test-bucket", credentials_path="test_credentials.json")

@pytest.fixture
def mock_callback():
    """Creates a mock callback function."""
    return Mock(return_value=FeedStatus.ACCEPT)

@pytest.fixture
def integration_setup(dummy_data_loader, mock_callback):
    """Creates a full integration setup with loaders and a FrameAnnotationLoader."""
    frame_annotation_loader = InstanceLoader(callback=mock_callback, buffer_size=1000)

    frame_loader = FrameLoader(dummy_data_loader, frame_annotation_loader.feed_frame,
                               frame_shape=(1080, 1920))

    annotation_loader = AnnotationLoader(dummy_data_loader, frame_annotation_loader.feed_annotation)

    return frame_annotation_loader, frame_loader, annotation_loader

def test_full_pipeline(integration_setup, mock_callback):
    """Tests if FrameLoader and AnnotationLoader correctly call FrameAnnotationLoader."""
    # arrange
    frame_annotation_loader, frame_loader, annotation_loader = integration_setup

    # act
    frame_loader.load_frames("test_video.mp4")
    annotation_loader.load_annotations("test_annotation.json")

    frame_loader.wait_for_completion()
    annotation_loader.wait_for_completion()

    # assert
    assert mock_callback.call_count == 172, "No frames were processed."