import cv2
import matplotlib.pyplot as plt

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


def manual_visualize_annotations():
    # --- Manual "fixture" setup ---
    auth_factory = GCPAuthServiceFactory(TestBucket.SERVICE_ACCOUNT_FILE)
    label_parser_factory = SimpleLabelParserFactory(NorsvinBehaviorClass.get_label_map())
    decoder_factory = DarwinDecoderFactory(label_parser_factory)
    loader_factory = GCSLoaderFactory(
        bucket_name=TestBucket.BUCKET_NAME,
        auth_factory=auth_factory,
        decoder_factory=decoder_factory
    )
    instance_provider = SimpleDatasetInstanceProvider(
        source=loader_factory.create_dataset_source(),
        video_selector=RandomFileSelector(["mp4"]),
        annotation_matcher=BaseNameMatchingStrategy(["json"])
    )
    entity_factory = LazyEntityFactory(loader_factory)
    frame_resizer_factory = StaticFrameResizerFactory((2688, 1520))
    bbox_normalizer_factory = SimpleBBoxNormalizerFactory((2688, 1520), (0, 1))
    streamer_pair_factory = FileStreamerPairFactory(
        instance_provider, entity_factory, frame_resizer_factory, bbox_normalizer_factory
    )
    source_id_provider = FileSourceIdProvider(loader_factory.create_dataset_source(), FileBaseNameParser())
    dataset_splitter = ConsistentDatasetSplitter()
    virtual_dataset = VirtualDataset(source_id_provider, dataset_splitter, max_sources=10, max_frames_per_source=200)
    source_parser_factory = FileBaseNameParserFactory()
    aggregated_streamer_factory = AggregatedStreamerFactory(
        streamer_pair_factory, callback=virtual_dataset.feed, source_parser_factory=source_parser_factory
    )

    # --- Start streaming ---
    streamer = aggregated_streamer_factory.create_streamer()
    streamer.start_streaming()
    streamer.wait_for_completion()

    # --- Visualization ---
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

        for annotation in annotations:
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


if __name__ == "__main__":
    manual_visualize_annotations()
