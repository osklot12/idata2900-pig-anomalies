import time
from typing import Dict

from src.data.dataset.selectors.factories.determ_string_selector_factory import DetermStringSelectorFactory
from src.data.dataset.selectors.factories.random_string_selector_factory import RandomStringSelectorFactory
from src.data.dataset.streams.factories.dock_stream_factory import DockStreamFactory
from src.data.dataset.streams.factories.pool_stream_factory import PoolStreamFactory
from src.data.dataset.streams.managed.factories.gcs_stream_factory import GCSStreamFactory
from src.data.pipeline.factories.norsvin_eval_pipeline_factory import NorsvinEvalPipelineFactory
from src.data.pipeline.factories.norsvin_train_pipeline_factory import NorsvinTrainPipelineFactory
from src.network.messages.requests.handlers.dataset_stream_factories import DatasetStreamFactories
from src.network.messages.requests.handlers.registry.factories.default_handler_registry_factory import \
    DefaultHandlerRegistryFactory
from src.network.messages.serialization.factories.pickle_deserializer_factory import PickleDeserializerFactory
from src.network.messages.serialization.factories.pickle_serializer_factory import PickleSerializerFactory
from src.network.server.network_server import NetworkServer
from src.network.server.session.factories.clean_session_factory import CleanSessionFactory
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataset.dataset_split import DatasetSplit
from src.utils.gcs_credentials import GCSCredentials
from src.utils.norsvin_dataset_config import NORSVIN_SPLIT_RATIOS
from tests.utils.gcs.test_bucket import TestBucket


def is_annotated(instance: AnnotatedFrame) -> bool:
    """Predicate for checking whether the instance is annotated."""
    return len(instance.annotations) > 0


def main():
    gcs_creds = GCSCredentials(bucket_name=TestBucket.BUCKET_NAME, service_account_path=TestBucket.SERVICE_ACCOUNT_FILE)
    split_ratios = NORSVIN_SPLIT_RATIOS

    def has_annotations(meta: Dict[str, int]) -> bool:
        return any(count > 0 for count in meta.values())

    train_stream_factory = GCSStreamFactory(
        gcs_creds=gcs_creds,
        split_ratios=split_ratios,
        split=DatasetSplit.TRAIN,
        selector_factory=RandomStringSelectorFactory(),
        label_map=NorsvinBehaviorClass.get_label_map(),
        stream_factory=PoolStreamFactory(pool_size=7000, min_ready=5000),
        pipeline_factory=NorsvinTrainPipelineFactory(),
        filter_func=has_annotations,
    )

    val_stream_factory = GCSStreamFactory(
        gcs_creds=gcs_creds,
        split_ratios=split_ratios,
        split=DatasetSplit.VAL,
        selector_factory=DetermStringSelectorFactory(),
        label_map=NorsvinBehaviorClass.get_label_map(),
        stream_factory=DockStreamFactory(buffer_size=3, dock_size=500),
        pipeline_factory=NorsvinEvalPipelineFactory()
    )

    test_stream_factory = GCSStreamFactory(
        gcs_creds=gcs_creds,
        split_ratios=split_ratios,
        split=DatasetSplit.TEST,
        selector_factory=DetermStringSelectorFactory(),
        label_map=NorsvinBehaviorClass.get_label_map(),
        stream_factory=DockStreamFactory(buffer_size=3, dock_size=500),
        pipeline_factory=NorsvinEvalPipelineFactory()
    )

    stream_factories = DatasetStreamFactories(
        train_factory=train_stream_factory,
        val_factory=val_stream_factory,
        test_factory=test_stream_factory
    )

    session_factory = CleanSessionFactory()
    handler_factory = DefaultHandlerRegistryFactory(stream_factories=stream_factories)

    server = NetworkServer(
        serializer_factory=PickleSerializerFactory(),
        deserializer_factory=PickleDeserializerFactory(),
        session_factory=session_factory,
        handler_factory=handler_factory
    )

    try:
        server.run()
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
