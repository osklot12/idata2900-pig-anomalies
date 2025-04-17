import pytest
import time

from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.selectors.factories.determ_string_selector_factory import DetermStringSelectorFactory
from src.data.dataset.streams.factories.dock_stream_factory import DockStreamFactory
from src.data.dataset.streams.managed.factories.gcs_stream_factory import GCSStreamFactory
from src.data.pipeline.factories.norsvin_train_pipeline_factory import NorsvinTrainPipelineFactory
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from src.utils.norsvin_dataset_config import NORSVIN_GCS_CREDS, NORSVIN_SPLIT_RATIOS


def test_norsvin_train_stream():
    """Tests that creating the Norsvin training set stream with GCSStreamFactory gives the correct stream."""
    # arrange
    stream_factory = GCSStreamFactory(
        gcs_creds=NORSVIN_GCS_CREDS,
        split_ratios=NORSVIN_SPLIT_RATIOS,
        split=DatasetSplit.TRAIN,
        selector_factory=DetermStringSelectorFactory(),
        label_map=NorsvinBehaviorClass.get_label_map(),
        stream_factory=DockStreamFactory(buffer_size=3, dock_size=1000),
        pipeline_factory=NorsvinTrainPipelineFactory()
    )

    stream = stream_factory.create_stream()

    # act
    stream.run()

    total = 0
    tail_biting = 0
    ear_biting = 0
    belly_nosing = 0
    tail_down = 0
    try:
        instance = stream.read()
        while instance:
            for annotation in instance.annotations:
                if annotation.cls == NorsvinBehaviorClass.TAIL_BITING:
                    tail_biting += 1
                if annotation.cls == NorsvinBehaviorClass.TAIL_DOWN:
                    tail_down += 1
                if annotation.cls == NorsvinBehaviorClass.BELLY_NOSING:
                    belly_nosing += 1
                if annotation.cls == NorsvinBehaviorClass.EAR_BITING:
                    ear_biting += 1
            total += 1
            print(f"[Test] Total instances: {total}")
            print(f"[Test] Bellynosing: {belly_nosing} ({belly_nosing/total*100:.2f}%)")
            print(f"[Test] Tailbiting: {tail_biting} ({tail_biting/total*100:.2f}%)")
            print(f"[Test] Earbiting: {ear_biting} ({ear_biting/total*100:.2f}%)")
            print(f"[Test] Taildown: {tail_down} ({tail_down/total*100:.2f}%)")
            instance = stream.read()
    except KeyboardInterrupt:
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
        assert isinstance(instance, AnnotatedFrame)
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
        assert isinstance(instance, AnnotatedFrame)
        # StreamedAnnotatedFrameVisualizer.visualize(instance)
        instance = stream.read()

    stream.stop()
