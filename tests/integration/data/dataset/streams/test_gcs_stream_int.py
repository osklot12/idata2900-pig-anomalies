import pytest

from src.data.dataclasses.dataset_split_ratios import DatasetSplitRatios
from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.managed.factories.gcs_stream_factory import GCSStreamFactory
from src.data.preprocessing.normalization.factories.bbox_normalizer_component_factory import \
    BBoxNormalizerComponentFactory
from src.data.preprocessing.normalization.factories.simple_bbox_normalizer_factory import SimpleBBoxNormalizerFactory
from src.data.preprocessing.resizing.factories.frame_resizer_component_factory import FrameResizerComponentFactory
from src.data.preprocessing.resizing.factories.static_frame_resizer_factory import StaticFrameResizerFactory
from src.utils.gcs_credentials import GCSCredentials
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from tests.utils.gcs.test_bucket import TestBucket
from tests.utils.streamed_annotated_frame_visualizer import StreamedAnnotatedFrameVisualizer


@pytest.fixture
def gcs_creds():
    """Fixture to provide a GCSCredentials instance."""
    return GCSCredentials(bucket_name=TestBucket.BUCKET_NAME, service_account_path=TestBucket.SERVICE_ACCOUNT_FILE)


@pytest.fixture
def split_ratios():
    """Fixture to provide a DatasetSplitRatios instance."""
    return DatasetSplitRatios(train=0.8, val=0.1, test=0.1)


@pytest.fixture
def resizer_component_factory():
    """Fixture to provide a FrameResizerComponentFactory instance."""
    frame_resizer_factory = StaticFrameResizerFactory((640, 640))
    return FrameResizerComponentFactory(frame_resizer_factory)


@pytest.fixture
def normalizer_component_factory():
    """Fixture to provide a BBoxNormalizerComponentFactory instance."""
    bbox_normalizer_factory = SimpleBBoxNormalizerFactory((0, 1))
    return BBoxNormalizerComponentFactory(bbox_normalizer_factory)


def test_norsvin_train_stream(gcs_creds, split_ratios, resizer_component_factory, normalizer_component_factory):
    """Tests that creating the Norsvin training set stream with GCSStreamFactory gives the correct stream."""
    # arrange
    stream_factory = GCSStreamFactory(
        gcs_creds=gcs_creds,
        split_ratios=split_ratios,
        split=DatasetSplit.TRAIN,
        label_map=NorsvinBehaviorClass.get_label_map(),
        preprocessor_factories=[resizer_component_factory, normalizer_component_factory]
    )
    stream = stream_factory.create_stream()

    # act
    stream.run()

    for _ in range(100):
        StreamedAnnotatedFrameVisualizer.visualize(stream.read())

    stream.stop()


def test_norsvin_val_stream(gcs_creds, split_ratios, resizer_component_factory, normalizer_component_factory):
    """Tests that creating the Norsvin validation set stream with GCSStreamFactory gives the correct stream."""
    # arrange
    stream_factory = GCSStreamFactory(
        gcs_creds=gcs_creds,
        split_ratios=split_ratios,
        split=DatasetSplit.VAL,
        label_map=NorsvinBehaviorClass.get_label_map(),
        preprocessor_factories=[resizer_component_factory, normalizer_component_factory]
    )

    stream = stream_factory.create_stream()

    # act
    stream.run()

    instance = stream.read()
    while instance:
        assert isinstance(instance, StreamedAnnotatedFrame)
        # StreamedAnnotatedFrameVisualizer.visualize(instance)
        instance = stream.read()

    stream.stop()


def test_norsvin_test_stream(gcs_creds, split_ratios, resizer_component_factory, normalizer_component_factory):
    """Tests that creating the Norsvin test set stream with GCSStreamFactory gives the correct stream."""
    # arrange
    stream_factory = GCSStreamFactory(
        gcs_creds=gcs_creds,
        split_ratios=split_ratios,
        split=DatasetSplit.TEST,
        label_map=NorsvinBehaviorClass.get_label_map(),
        preprocessor_factories=[resizer_component_factory, normalizer_component_factory]
    )

    stream = stream_factory.create_stream()

    # act
    stream.run()

    instance = stream.read()
    while instance:
        assert isinstance(instance, StreamedAnnotatedFrame)
        StreamedAnnotatedFrameVisualizer.visualize(instance)
        instance = stream.read()

    stream.stop()
