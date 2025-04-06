import cv2
import matplotlib.pyplot as plt

from src.auth.factories.gcp_auth_service_factory import GCPAuthServiceFactory
from src.data.dataset.factories.lazy_entity_factory import LazyEntityFactory
from src.data.dataset.matching.base_name_matcher import BaseNameMatcher
from src.data.dataset.providers.simple_dataset_instance_provider import SimpleDatasetInstanceProvider
from src.data.dataset.selection.random_file_selector import RandomFileSelector
from src.data.dataset.splitters.consistent_dataset_splitter import ConsistentDatasetSplitter
from src.data.dataset.virtual.virtual_dataset import VirtualDataset
from src.data.dataset.dataset_split import DatasetSplit
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


def manual_visualize_annotations():
    # setup
    loader_factory = _get_loader_factory()

    instance_provider = _get_instance_provider(loader_factory)

    streamer_pair_factory = _get_streamer_pair_factory(instance_provider, loader_factory)

    virtual_dataset = _get_virtual_dataset(loader_factory)

    aggregated_streamer_factory = _get_aggregated_streamer(streamer_pair_factory, virtual_dataset)

    # streams
    _stream_frames(aggregated_streamer_factory)

    # get batch
    split_buffer = _get_non_empty_internal_buffer(virtual_dataset)
    key = split_buffer.keys()[0]
    source_buffer = split_buffer.at(key)

    # visualize
    for frame_index in source_buffer.keys():
        instance = source_buffer.at(frame_index)
        frame = instance.frame
        annotations = instance.annotations
        print(f"{annotations}")
        frame_height, frame_width, _ = frame.shape

        for annotation in annotations:
            behavior = annotation.cls
            print(f"Stored annotation {frame_index}: {annotation}")
            x_min, y_min, x_max, y_max = _get_absolute_coordinates(annotation, frame_width, frame_height)
            print(
                f"Absolute coordinates: x({(x_min + x_max) / 2}), y({(y_min + y_max) / 2}), w({x_max - x_min}), h({y_max - y_min})")

            _draw_bbox(behavior, frame, x_max, x_min, y_max, y_min)

        _show_plot(frame, frame_index)


def _show_plot(frame, frame_index):
    plt.figure(figsize=(6, 6))
    plt.imshow(frame)
    plt.title(f"Frame {frame_index}")
    plt.axis("off")
    plt.show()


def _draw_bbox(behavior, frame, x_max, x_min, y_max, y_min):
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    cv2.putText(
        frame, behavior.name.replace("_", " "), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 255, 0), 1, cv2.LINE_AA
    )


def _get_absolute_coordinates(annotation, frame_width, frame_height):
    x = annotation.bbox.x
    y = annotation.bbox.y
    w = annotation.bbox.width
    h = annotation.bbox.height

    # flip y axis for opencv
    # y = 1.0 - y

    x_min = int(x * frame_width)
    x_max = int((x + w) * frame_width)
    y_min = int(y * frame_height)
    y_max = int((y + h) * frame_height)

    return x_min, y_min, x_max, y_max


def _get_non_empty_internal_buffer(virtual_dataset):
    if virtual_dataset.get_frame_count(DatasetSplit.TRAIN):
        frames = virtual_dataset._buffer_map.get(DatasetSplit.TRAIN)
    elif virtual_dataset.get_frame_count(DatasetSplit.VAL):
        frames = virtual_dataset._buffer_map.get(DatasetSplit.VAL)
    else:
        frames = virtual_dataset._buffer_map.get(DatasetSplit.TEST)
    return frames


def _stream_frames(aggregated_streamer_factory):
    streamer = aggregated_streamer_factory.create_streamer()
    streamer.start_streaming()
    streamer.wait_for_completion()


def _get_aggregated_streamer(streamer_pair_factory, virtual_dataset):
    source_parser_factory = FileBaseNameParserFactory()
    aggregated_streamer_factory = AggregatedStreamerFactory(
        streamer_pair_factory=streamer_pair_factory,
        callback=virtual_dataset.feed,
        source_parser_factory=source_parser_factory
    )
    return aggregated_streamer_factory


def _get_virtual_dataset(loader_factory):
    dataset_splitter = ConsistentDatasetSplitter()
    virtual_dataset = VirtualDataset(dataset_splitter, max_sources=10, max_frames_per_source=1000)
    return virtual_dataset


def _get_streamer_pair_factory(instance_provider, loader_factory):
    entity_factory = LazyEntityFactory(loader_factory, FileBaseNameParser())
    frame_resizer_factory = StaticFrameResizerFactory((640, 640))
    bbox_normalizer_factory = SimpleBBoxNormalizerFactory((0, 1))
    streamer_pair_factory = FileStreamerPairFactory(
        instance_provider, entity_factory, frame_resizer_factory, bbox_normalizer_factory
    )
    return streamer_pair_factory


def _get_instance_provider(loader_factory):
    instance_provider = SimpleDatasetInstanceProvider(
        source=loader_factory.create_dataset_source(),
        video_selector=RandomFileSelector(["mp4"]),
        annotation_matcher=BaseNameMatcher(["json"])
    )
    return instance_provider


def _get_loader_factory():
    auth_factory = GCPAuthServiceFactory(TestBucket.SERVICE_ACCOUNT_FILE)
    label_parser_factory = SimpleLabelParserFactory(NorsvinBehaviorClass.get_label_map())
    decoder_factory = DarwinDecoderFactory(label_parser_factory)
    loader_factory = GCSLoaderFactory(
        bucket_name=TestBucket.BUCKET_NAME,
        auth_factory=auth_factory,
        decoder_factory=decoder_factory
    )
    return loader_factory


if __name__ == "__main__":
    manual_visualize_annotations()
