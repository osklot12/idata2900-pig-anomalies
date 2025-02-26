import time
from unittest.mock import Mock

import pytest

from src.auth.gcp_auth_service import GCPAuthService
from src.data.loading.annotation_loader import AnnotationLoader
from src.data.loading.frame_annotation_loader import FrameAnnotationLoader
from src.data.loading.frame_loader import FrameLoader
from src.data.loading.gcp_data_loader import GCPDataLoader
from src.data.virtual_dataset import VirtualDataset
from src.utils.norsvin_bucket_parser import NorsvinBucketParser
from src.utils.source_normalizer import SourceNormalizer
from tests.utils.constants.sample_bucket_files import SampleBucketFiles


@pytest.fixture
def frame_data_loader():
    """Returns a GCPDataLoader instance frame loading."""
    auth_service = GCPAuthService(credentials_path=NorsvinBucketParser.CREDENTIALS_PATH)
    return GCPDataLoader(bucket_name=NorsvinBucketParser.BUCKET_NAME, auth_service=auth_service)

@pytest.fixture
def annotation_data_loader():
    """Returns a GCPDataLoader instance for annotation loading."""
    auth_service = GCPAuthService(credentials_path=NorsvinBucketParser.CREDENTIALS_PATH)
    return GCPDataLoader(bucket_name=NorsvinBucketParser.BUCKET_NAME, auth_service=auth_service)

@pytest.fixture
def dataset_id_loader():
    """Returns a GCPDataLoader instance for dataset id loading."""
    auth_service = GCPAuthService(credentials_path=NorsvinBucketParser.CREDENTIALS_PATH)
    return GCPDataLoader(bucket_name=NorsvinBucketParser.BUCKET_NAME, auth_service=auth_service)

@pytest.fixture
def virtual_dataset(dataset_id_loader):
    """Returns a VirtualDataset instance."""
    dataset_ids = dataset_id_loader.list_files(NorsvinBucketParser.VIDEO_PREFIX, "")
    dataset_ids = [SourceNormalizer.normalize(id) for id in dataset_ids]
    return VirtualDataset(dataset_ids, max_sources=10, max_frames_per_source=300)

@pytest.fixture
def mock_callback():
    """Fixture to provide a mock callback."""
    return Mock()

@pytest.mark.integration
def test_streaming_data(frame_data_loader, annotation_data_loader, virtual_dataset, mock_callback):
    """Tests that data is successfully streamed from the could and stored in the VirtualDataset."""
    # arrange
    frame_annotation_loader = FrameAnnotationLoader(virtual_dataset.feed)

    frame_loader = FrameLoader(
        data_loader=frame_data_loader,
        callback=frame_annotation_loader.feed_frame,
        frame_shape=(1520, 2688),
        resize_shape=(640, 640)
    )

    annotation_loader = AnnotationLoader(
        data_loader=annotation_data_loader,
        callback=frame_annotation_loader.feed_annotation
    )

    # act
    frame_loader.load_frames(NorsvinBucketParser.get_video_blob_name(SampleBucketFiles.SAMPLE_VIDEO_FILE))
    annotation_loader.load_annotations(NorsvinBucketParser.get_annotation_blob_name(SampleBucketFiles.SAMPLE_JSON_FILE))

    frame_loader.wait_for_completion()
    annotation_loader.wait_for_completion()

    # assert
    sample_source = SourceNormalizer.normalize(SampleBucketFiles.SAMPLE_VIDEO_FILE)
    split_buffer = virtual_dataset._get_buffer_for_source(sample_source)

    assert split_buffer.has(sample_source)
    assert split_buffer.at(sample_source).size() == 208