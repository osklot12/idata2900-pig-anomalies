from unittest.mock import Mock
import cv2
import matplotlib.pyplot as plt

import pytest

from src.auth.factories.gcp_auth_service_factory import GCPAuthServiceFactory
from src.data.dataset.factories.lazy_entity_factory import LazyEntityFactory
from src.data.dataset.matching.base_name_matching_strategy import BaseNameMatchingStrategy
from src.data.dataset.providers.file_source_id_provider import FileSourceIdProvider
from src.data.dataset.providers.simple_dataset_instance_provider import SimpleDatasetInstanceProvider
from src.data.dataset.selection.random_file_selector import RandomFileSelector
from src.data.dataset.splitters.consistent_dataset_splitter import ConsistentDatasetSplitter
from src.data.dataset.virtual_dataset import VirtualDataset
from src.data.dataset_split import DatasetSplit
from src.data.decoders.factories.darwin_decoder_factory import DarwinDecoderFactory
from src.data.label.factories.simple_label_parser_factory import SimpleLabelParserFactory
from src.data.loading.factories.gcs_loader_factory import GCSLoaderFactory
from src.data.parsing.factories.FileBaseNameParserFactory import FileBaseNameParserFactory
from src.data.parsing.file_base_name_parser import FileBaseNameParser
from src.data.preprocessing.normalization.factories.simple_bbox_normalizer_factory import SimpleBBoxNormalizerFactory
from src.data.preprocessing.resizing.factories.static_frame_resizer_factory import StaticFrameResizerFactory
from src.data.streaming.factories.aggregated_streamer_factory import AggregatedStreamerFactory
from src.data.streaming.factories.file_streamer_pair_factory import FileStreamerPairFactory
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from tests.utils.gcs.test_bucket import TestBucket


@pytest.fixture
def auth_factory():
    """Fixture to provide a AuthServiceFactory instance."""
    return GCPAuthServiceFactory(TestBucket.SERVICE_ACCOUNT_FILE)


@pytest.fixture
def label_parser_factory():
    """Fixture to provide a LabelParserFactory instance."""
    return SimpleLabelParserFactory(NorsvinBehaviorClass.get_label_map())


@pytest.fixture
def decoder_factory(label_parser_factory):
    """Fixture to provide a DecoderFactory instance."""
    return DarwinDecoderFactory(label_parser_factory)


@pytest.fixture
def loader_factory(auth_factory, decoder_factory):
    """Fixture to provide a LoaderFactory instance."""
    return GCSLoaderFactory(
        bucket_name=TestBucket.BUCKET_NAME,
        auth_factory=auth_factory,
        decoder_factory=decoder_factory
    )


@pytest.fixture
def instance_provider(loader_factory):
    """Fixture to provide a DatasetInstanceProvider instance."""
    return SimpleDatasetInstanceProvider(
        source=loader_factory.create_dataset_source(),
        video_selector=RandomFileSelector(["mp4"]),
        annotation_matcher=BaseNameMatchingStrategy(["json"])
    )


@pytest.fixture
def entity_factory(loader_factory):
    """Fixture to provide a DatasetEntityFactory instance."""
    return LazyEntityFactory(loader_factory)


@pytest.fixture
def frame_resizer_factory():
    """Fixture to provide a FrameResizerFactory instance."""
    return StaticFrameResizerFactory((2688, 1520))


@pytest.fixture
def bbox_normalizer_factory():
    """Fixture to provide a BBoxNormalizerFactory instance."""
    return SimpleBBoxNormalizerFactory((2688, 1520), (0, 1))


@pytest.fixture
def streamer_pair_factory(instance_provider, entity_factory, frame_resizer_factory, bbox_normalizer_factory):
    """Fixture to provide a StreamerPairFactory instance."""
    return FileStreamerPairFactory(
        instance_provider=instance_provider,
        entity_factory=entity_factory,
        frame_resizer_factory=frame_resizer_factory,
        bbox_normalizer_factory=bbox_normalizer_factory
    )


@pytest.fixture
def source_id_provider(loader_factory):
    """Fixture to provide a SourceIdProvider instance."""
    return FileSourceIdProvider(loader_factory.create_dataset_source(), FileBaseNameParser())


@pytest.fixture
def dataset_splitter():
    """Fixture to provide a DatasetSplitter instance."""
    return ConsistentDatasetSplitter()


@pytest.fixture
def virtual_dataset(source_id_provider, dataset_splitter):
    """Fixture to provide a VirtualDataset instance."""
    return VirtualDataset(
        id_provider=source_id_provider,
        splitter=dataset_splitter,
        max_sources=10,
        max_frames_per_source=200
    )


@pytest.fixture
def source_parser_factory():
    """Fixture to provide a StringParserFactory instance."""
    return FileBaseNameParserFactory()


@pytest.fixture
def aggregated_streamer_factory(streamer_pair_factory, virtual_dataset, source_parser_factory):
    """Fixture to provide an AggregatedStreamerFactory instance."""
    return AggregatedStreamerFactory(
        streamer_pair_factory=streamer_pair_factory,
        callback=virtual_dataset.feed,
        source_parser_factory=source_parser_factory
    )


@pytest.mark.integration
def test_streaming_data(virtual_dataset, aggregated_streamer_factory):
    """Tests that data is successfully streamed from the could and stored in the VirtualDataset."""
    # arrange
    streamer = aggregated_streamer_factory.create_streamer()

    # act
    streamer.start_streaming()
    streamer.wait_for_completion()


@pytest.mark.integration
def test_visualize_annotations(virtual_dataset, aggregated_streamer_factory):
    """Tests visualization of streaming data with bounding boxes."""
    # arrange
    streamer = aggregated_streamer_factory.create_streamer()

    # act
    streamer.start_streaming()
    streamer.wait_for_completion()

    # assert
    shuffled_batch = None

    batch_size = 40
    if virtual_dataset.get_frame_count(DatasetSplit.TRAIN):
        shuffled_batch = virtual_dataset.get_shuffled_batch(DatasetSplit.TRAIN, batch_size)
    elif virtual_dataset.get_frame_count(DatasetSplit.VAL):
        shuffled_batch = virtual_dataset.get_shuffled_batch(DatasetSplit.VAL, batch_size)
    else:
        shuffled_batch = virtual_dataset.get_shuffled_batch(DatasetSplit.TEST, batch_size)

    for instance in shuffled_batch:
        frame = instance.frame.copy()
        annotations = instance.annotations.copy()
        frame_height, frame_width, _ = frame.shape

        if annotations:
            for annotation in annotations:
                print(f"{annotation}")
                behavior = annotation.cls
                x = annotation.bbox.center_x
                y = annotation.bbox.center_y
                w = annotation.bbox.width
                h = annotation.bbox.height
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
        plt.title(f"Shuffled Frame")
        plt.axis("off")
        plt.show()
