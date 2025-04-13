import pytest

from src.data.dataclasses.dataset_split_ratios import DatasetSplitRatios
from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.factories.gcs_stream_factory import GCSStreamFactory
from src.data.preprocessing.normalization.factories.bbox_normalizer_component_factory import \
    BBoxNormalizerComponentFactory
from src.data.preprocessing.normalization.factories.simple_bbox_normalizer_factory import SimpleBBoxNormalizerFactory
from src.data.preprocessing.resizing.factories.frame_resizer_component_factory import FrameResizerComponentFactory
from src.data.preprocessing.resizing.factories.static_frame_resizer_factory import StaticFrameResizerFactory
from src.utils.gcs_credentials import GCSCredentials
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from tests.utils.gcs.test_bucket import TestBucket
from tests.utils.streamed_annotated_frame_visualizer import StreamedAnnotatedFrameVisualizer


def test_norsvin_train_stream():
    """Tests that creating the Norsvin training set stream with GCSStreamFactory gives a working stream."""
    # arrange
    gcs_creds = GCSCredentials(bucket_name=TestBucket.BUCKET_NAME, service_account_path=TestBucket.SERVICE_ACCOUNT_FILE)
    split_ratios = DatasetSplitRatios(train=0.8, val=0.1, test=0.1)

    frame_resizer_factory = StaticFrameResizerFactory((1920, 1080))
    resizer_component_factory = FrameResizerComponentFactory(frame_resizer_factory)

    bbox_normalizer_factory = SimpleBBoxNormalizerFactory((0, 1))
    normalizer_component_factory = BBoxNormalizerComponentFactory(bbox_normalizer_factory)

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

    for _ in range(10):
        StreamedAnnotatedFrameVisualizer.visualize(stream.read())

    stream.stop()