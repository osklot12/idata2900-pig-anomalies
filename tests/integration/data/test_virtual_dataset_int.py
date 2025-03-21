from unittest.mock import Mock
import cv2
import matplotlib.pyplot as plt

import pytest

from src.auth.gcp_auth_service import GCPAuthService
from src.data.preprocessing.normalization.normalizers.simple_bbox_normalizer import SimpleBBoxNormalizer
from src.data.decoders.darwin_decoder import DarwinDecoder
from src.data.streaming.streamers import AnnotationStreamer
from src.data.streaming.aggregators.buffered_instance_aggregator import BufferedInstanceAggregator
from src.data.streaming.streamers import FrameStreamer
from src.data.loading.loaders.gcs_data_loader_old import GCSDataLoaderOld
from src.data.dataset.virtual_dataset import VirtualDataset
from src.utils.norsvin.norsvin_annotation_parser import NorsvinLabelParser
from src.utils.norsvin_bucket_parser import NorsvinBucketParser
from src.utils.source_normalizer import SourceNormalizer
from tests.utils.constants.sample_bucket_files import SampleBucketFiles


@pytest.fixture
def frame_data_loader():
    """Returns a GCPDataLoader instance frame loading."""
    auth_service = GCPAuthService(credentials_path=NorsvinBucketParser.CREDENTIALS_PATH)
    return GCSDataLoaderOld(bucket_name=NorsvinBucketParser.BUCKET_NAME, auth_service=auth_service)

@pytest.fixture
def annotation_data_loader():
    """Returns a GCPDataLoader instance for annotation loading."""
    auth_service = GCPAuthService(credentials_path=NorsvinBucketParser.CREDENTIALS_PATH)
    return GCSDataLoaderOld(bucket_name=NorsvinBucketParser.BUCKET_NAME, auth_service=auth_service)

@pytest.fixture
def dataset_id_loader():
    """Returns a GCPDataLoader instance for dataset id loading."""
    auth_service = GCPAuthService(credentials_path=NorsvinBucketParser.CREDENTIALS_PATH)
    return GCSDataLoaderOld(bucket_name=NorsvinBucketParser.BUCKET_NAME, auth_service=auth_service)

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

@pytest.fixture
def setup_loaders(virtual_dataset, frame_data_loader, annotation_data_loader):
    """Sets up and returns loaders for testing."""
    instance_loader = BufferedInstanceAggregator(virtual_dataset.feed)

    frame_loader = FrameStreamer(
        data_loader=frame_data_loader,
        video_blob_name=NorsvinBucketParser.get_video_blob_name(SampleBucketFiles.SAMPLE_VIDEO_FILE),
        callback=instance_loader.feed_frame,
        frame_shape=(1520, 2688),
        resize_shape=(640, 640)
    )

    normalizer = SimpleBBoxNormalizer(image_dimensions=(2688, 1520), new_range=(0, 1),
                                      annotation_parser=NorsvinLabelParser)
    annotation_loader = AnnotationStreamer(
        data_loader=annotation_data_loader,
        annotation_blob_name=NorsvinBucketParser.get_annotation_blob_name(SampleBucketFiles.SAMPLE_JSON_FILE),
        decoder_cls=DarwinDecoder,
        normalizer=normalizer,
        callback=instance_loader.feed_annotation
    )

    return instance_loader, frame_loader, annotation_loader

@pytest.mark.integration
def test_streaming_data(frame_data_loader, annotation_data_loader, virtual_dataset, mock_callback, setup_loaders):
    """Tests that data is successfully streamed from the could and stored in the VirtualDataset."""
    # arrange
    instance_loader, frame_loader, annotation_loader = setup_loaders

    # act
    frame_loader.start_streaming()
    annotation_loader.start_streaming()

    frame_loader.wait_for_completion()
    annotation_loader.wait_for_completion()

    # assert
    sample_source = SourceNormalizer.normalize(SampleBucketFiles.SAMPLE_VIDEO_FILE)
    split_buffer = virtual_dataset._get_buffer_for_source(sample_source)

    assert split_buffer.has(sample_source)
    assert split_buffer.at(sample_source).size() == 208


@pytest.mark.integration
def test_visualize_annotations(frame_data_loader, annotation_data_loader, virtual_dataset, setup_loaders):
    """Tests visualization of streaming data with bounding boxes."""
    # arrange
    instance_loader, frame_loader, annotation_loader = setup_loaders

    # act
    frame_loader.start_streaming()
    annotation_loader.start_streaming()

    frame_loader.wait_for_completion()
    annotation_loader.wait_for_completion()

    # assert
    sample_source = SourceNormalizer.normalize(SampleBucketFiles.SAMPLE_VIDEO_FILE)
    split_buffer = virtual_dataset._get_buffer_for_source(sample_source)

    assert split_buffer.has(sample_source)
    assert split_buffer.at(sample_source).size() == 208

    batch_size = 5
    split = virtual_dataset.get_split_for_source(sample_source)
    shuffled_batch = virtual_dataset.get_shuffled_batch(split, batch_size)

    assert len(shuffled_batch) == batch_size

    for i, (frame, annotations) in enumerate(shuffled_batch):
        frame = frame.copy()
        frame_height, frame_width, _ = frame.shape

        if annotations:
            for behavior, x, y, w, h in annotations:
                x_min = int((x - w / 2) * frame_width)
                y_min = int((y - h / 2) * frame_height)
                x_max = int((x + w / 2) * frame_width)
                y_max = int((y + h / 2) * frame_height)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(
                    frame, behavior.name.replace("_", " "), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA
                )

        plt.figure(figsize=(6, 6))
        plt.imshow(frame)
        plt.title(f"Shuffled Frame {i + 1}")
        plt.axis("off")
        plt.show()