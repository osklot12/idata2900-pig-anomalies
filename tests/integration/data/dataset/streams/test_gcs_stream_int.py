import pytest
import time

from src.data.dataset.streams.managed.factories.gcs_stream_factory import GCSStreamFactory
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from src.utils.norsvin_dataset_config import NORSVIN_GCS_CREDS, NORSVIN_SPLIT_RATIOS


def test_norsvin_train_stream():
    """Tests that creating the Norsvin training set stream with GCSStreamFactory gives the correct stream."""
    # arrange
    stream_factory = GCSStreamFactory(
        gcs_creds=NORSVIN_GCS_CREDS,
        split_ratios=NORSVIN_SPLIT_RATIOS,
        split=
        label_map=NorsvinBehaviorClass.get_label_map(),
        stream_factory=
    )

    stream = stream_factory.create_stream()

    # act
    stream.run()

    # for _ in range(1000000):
    # stream.read()
    # StreamedAnnotatedFrameVisualizer.visualize(stream.read())
    running = AtomicBool(True)
    def print_memory():
        while running:
            print(report_memory())
            time.sleep(0.5)

    t = threading.Thread(target=print_memory)

    total = 0
    tail_biting = 0
    ear_biting = 0
    belly_nosing = 0
    tail_down = 0
    try:
        # t.start()
        instance = stream.read()
        while instance:
            StreamedAnnotatedFrameVisualizer.visualize(instance)
            print(f"[Test] Got item: {type(instance)}")
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
        running.set(False)
        stream.stop()
        t.join()


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
