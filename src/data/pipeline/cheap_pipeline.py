from typing import List

from src.auth.factories.gcp_auth_service_factory import GCPAuthServiceFactory
from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataset.factories.lazy_entity_factory import LazyEntityFactory
from src.data.dataset.matching.base_name_matching_strategy import BaseNameMatchingStrategy
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
from src.data.streaming.managers.docking_streamer_manager import DockingStreamerManager
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from tests.utils.gcs.test_bucket import TestBucket


class CheapPipeline:
    """A cheap pipeline facade for quick training."""

    def __init__(self):
        # loader factory setup
        auth_factory = GCPAuthServiceFactory(TestBucket.SERVICE_ACCOUNT_FILE)

        decoder_factory = DarwinDecoderFactory(SimpleLabelParserFactory(NorsvinBehaviorClass.get_label_map()))

        self._loader_factory = GCSLoaderFactory(
            bucket_name=TestBucket.BUCKET_NAME,
            auth_factory=auth_factory,
            decoder_factory=decoder_factory
        )

        # streamer factory setup
        instance_provider = SimpleDatasetInstanceProvider(
            source=self._loader_factory.create_dataset_source(),
            video_selector=RandomFileSelector(["mp4"]),
            annotation_matcher=BaseNameMatchingStrategy(["json"])
        )

        entity_factory = LazyEntityFactory(self._loader_factory, FileBaseNameParser())

        frame_resizer_factory = StaticFrameResizerFactory((640, 640))

        bbox_normalizer_factory = SimpleBBoxNormalizerFactory((0, 1))

        streamer_pair_provider = FileStreamerPairFactory(
            instance_provider=instance_provider,
            entity_factory=entity_factory,
            frame_resizer_factory=frame_resizer_factory,
            bbox_normalizer_factory=bbox_normalizer_factory
        )

        self._virtual_dataset = VirtualDataset(
            splitter=ConsistentDatasetSplitter(),
            max_sources=10,
            max_frames_per_source=1000
        )

        self._aggregated_streamer_factory = AggregatedStreamerFactory(
            streamer_pair_factory=streamer_pair_provider,
            callback=self._virtual_dataset.feed,
            source_parser_factory=FileBaseNameParserFactory()
        )

        # streamer manager setup
        self._streamer_manager = DockingStreamerManager(
            streamer_provider=self._aggregated_streamer_factory,
            n_streamers=4
        )

    def run(self) -> None:
        """Runs the pipeline."""
        self._streamer_manager.run()

    def stop(self) -> None:
        """Stops the pipeline."""
        self._streamer_manager.stop()

    def get_batch(self, split: DatasetSplit, batch_size: int) -> List[AnnotatedFrame]:
        """
        Returns a batch of data.

        Args:
            split (DatasetSplit): the split to get from
            batch_size (int): the batch size

        Returns:
            List[AnnotatedFrame]: the batch
        """
        return self._virtual_dataset.get_shuffled_batch(split, batch_size)