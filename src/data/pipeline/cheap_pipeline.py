from typing import List

from src.auth.factories.gcp_auth_service_factory import GCPAuthServiceFactory
from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataset.factories.lazy_entity_factory import LazyEntityFactory
from src.data.dataset.matching.base_name_matcher import BaseNameMatcher
from src.data.dataset.providers.simple_dataset_instance_provider import SimpleDatasetInstanceProvider
from src.data.dataset.selectors.random_string_selector import RandomStringSelector
from src.data.dataset.splitters.consistent_dataset_splitter import ConsistentDatasetSplitter
from src.data.dataset.virtual.frame_dataset import FrameDataset
from src.data.dataset.dataset_split import DatasetSplit
from src.data.decoders.factories.darwin_decoder_factory import DarwinDecoderFactory
from src.data.label.factories.simple_label_parser_factory import SimpleLabelParserFactory
from src.data.loading.factories.gcs_loader_factory import GCSLoaderFactory
from src.data.parsing.factories.FileBaseNameParserFactory import FileBaseNameParserFactory
from src.data.parsing.base_name_parser import BaseNameParser
from src.data.preprocessing.normalization.factories.simple_bbox_normalizer_factory import SimpleBBoxNormalizerFactory
from src.data.preprocessing.resizing.factories.static_frame_resizer_factory import StaticFrameResizerFactory
from src.data.providers.batch_provider import BatchProvider
from src.data.streaming.factories.aggregated_streamer_factory import AggregatedStreamerFactory
from src.data.streaming.factories.file_streamer_pair_factory import FileStreamerPairFactory
from src.data.streaming.managers.dynamic_streamer_manager import DynamicStreamerManager
from src.schemas.aggregators.pressure_to_metric import PressureToMetric
from src.schemas.algorithms.simple_demand_estimator import SimpleDemandEstimator
from src.schemas.brokers.schema_broker import SchemaBroker
from src.schemas.signing.sign import Sign
from src.schemas.signing.simple_schema_signer import SimpleSchemaSigner
from src.ui.telemetry.rich_dashboard import RichDashboard
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from tests.utils.gcs.test_bucket import TestBucket


class CheapPipeline(BatchProvider):
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
            video_selector=RandomStringSelector(["mp4"]),
            annotation_matcher=BaseNameMatcher(["json"])
        )

        entity_factory = LazyEntityFactory(self._loader_factory, BaseNameParser())

        frame_resizer_factory = StaticFrameResizerFactory((640, 640))

        bbox_normalizer_factory = SimpleBBoxNormalizerFactory((0, 1))

        streamer_pair_provider = FileStreamerPairFactory(
            instance_provider=instance_provider,
            entity_factory=entity_factory,
            frame_resizer_factory=frame_resizer_factory,
            bbox_normalizer_factory=bbox_normalizer_factory
        )

        pressure_broker = SchemaBroker()

        self._virtual_dataset = FrameDataset(
            splitter=ConsistentDatasetSplitter(train_ratio=0.6, val_ratio=0.2),
            max_size=5000,
            pressure_broker=pressure_broker
        )

        self._dashboard = RichDashboard()
        pressure_broker.subscribe(PressureToMetric(Sign(self._dashboard, SimpleSchemaSigner("vdataset"))))

        self._aggregated_streamer_factory = AggregatedStreamerFactory(
            streamer_pair_factory=streamer_pair_provider,
            callback=self._virtual_dataset.feed,
            source_parser_factory=FileBaseNameParserFactory()
        )

        # streamer manager setup
        self._streamer_manager = DynamicStreamerManager(
            streamer_factory=self._aggregated_streamer_factory,
            min_streamers=0,
            max_streamers=10,
            demand_estimator=SimpleDemandEstimator()
        )

        pressure_broker.subscribe(self._streamer_manager)

    def run(self) -> None:
        """Runs the pipeline."""
        self._dashboard.console.log("Running pipeline...")
        self._dashboard.start()
        self._streamer_manager.run()

    def stop(self) -> None:
        """Stops the pipeline."""
        self._streamer_manager.stop()
        self._dashboard.stop()

    def get_batch(self, split: DatasetSplit, batch_size: int) -> List[AnnotatedFrame]:
        return self._virtual_dataset.get_batch(split, batch_size)