from unittest.mock import Mock

import pytest

from src.data.preprocessing.normalization.normalizers.simple_bbox_normalizer import SimpleBBoxNormalizer
from src.data.decoders.darwin_decoder import DarwinDecoder
from src.data.streaming.streamers import AnnotationStreamer
from src.data.loading.feed_status import FeedStatus
from src.data.streaming.aggregators.buffered_instance_aggregator import BufferedInstanceAggregator
from src.data.streaming.streamers import FrameStreamer
from src.utils.norsvin.norsvin_annotation_parser import NorsvinLabelParser
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
    frame_annotation_loader = BufferedInstanceAggregator(callback=mock_callback, buffer_size=1000)

    frame_loader = FrameStreamer(
        data_loader=dummy_data_loader,
        video_blob_name="test-video",
        callback=frame_annotation_loader.feed_frame,
        frame_shape=(1080, 1920)
    )

    normalizer = SimpleBBoxNormalizer(image_dimensions=(1920, 1080), new_range=(0, 1),
                                      annotation_parser=NorsvinLabelParser)
    annotation_loader = AnnotationStreamer(
        data_loader=dummy_data_loader,
        annotation_blob_name="test_annotations.json",
        decoder_cls=DarwinDecoder,
        normalizer=normalizer,
        callback=mock_callback
    )

    return frame_annotation_loader, frame_loader, annotation_loader


def test_instance_loader(integration_setup, mock_callback):
    """Tests if FrameLoader and AnnotationLoader correctly call FrameAnnotationLoader."""
    # arrange
    frame_annotation_loader, frame_loader, annotation_loader = integration_setup

    # act
    frame_loader.start_streaming()
    annotation_loader.start_streaming()

    frame_loader.wait_for_completion()
    annotation_loader.wait_for_completion()

    # assert
    assert mock_callback.call_count == 172, "No frames were processed."
