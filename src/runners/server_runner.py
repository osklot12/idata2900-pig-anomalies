import time

from src.data.dataset.streams.managed.factories.norsvin_stream_factory import NorsvinStreamFactory
from src.data.preprocessing.normalization.factories.bbox_normalizer_component_factory import \
    BBoxNormalizerComponentFactory
from src.data.preprocessing.normalization.factories.simple_bbox_normalizer_factory import SimpleBBoxNormalizerFactory
from src.data.preprocessing.resizing.factories.frame_resizer_component_factory import FrameResizerComponentFactory
from src.data.preprocessing.resizing.factories.static_frame_resizer_factory import StaticFrameResizerFactory
from src.network.messages.requests.handlers.registry.factories.default_handler_registry_factory import \
    DefaultHandlerRegistryFactory
from src.network.messages.serialization.factories.pickle_deserializer_factory import PickleDeserializerFactory
from src.network.messages.serialization.factories.pickle_serializer_factory import PickleSerializerFactory
from src.network.server.network_server import NetworkServer
from src.network.server.session.factories.clean_session_factory import CleanSessionFactory
from src.utils.gcs_credentials import GCSCredentials
from src.utils.norsvin_dataset_config import NORSVIN_SPLIT_RATIOS
from tests.utils.gcs.test_bucket import TestBucket


def main():
    gcs_creds = GCSCredentials(bucket_name=TestBucket.BUCKET_NAME, service_account_path=TestBucket.SERVICE_ACCOUNT_FILE)
    split_ratios = NORSVIN_SPLIT_RATIOS
    resizer_factory = FrameResizerComponentFactory(StaticFrameResizerFactory((640, 640)))
    normalizer_factory = BBoxNormalizerComponentFactory(SimpleBBoxNormalizerFactory((0, 1)))
    stream_factory = NorsvinStreamFactory(
        gcs_creds=gcs_creds,
        split_ratios=split_ratios,
        preprocessor_factories=[resizer_factory, normalizer_factory]
    )

    session_factory = CleanSessionFactory()
    handler_factory = DefaultHandlerRegistryFactory(stream_factory=stream_factory)

    server = NetworkServer(
        serializer_factory=PickleSerializerFactory(),
        deserializer_factory=PickleDeserializerFactory(),
        session_factory=session_factory,
        handler_factory=handler_factory
    )

    server.run()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        server.stop()

if __name__ == "__main__":
    main()