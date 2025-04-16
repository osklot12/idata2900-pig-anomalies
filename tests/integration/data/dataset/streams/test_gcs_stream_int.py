import pytest
import time

from src.data.dataclasses.dataset_split_ratios import DatasetSplitRatios
from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.managed.factories.gcs_stream_factory import GCSStreamFactory
from src.data.preprocessing.augmentation.augmentors.factories.augmentor_component_factory import \
    AugmentorComponentFactory
from src.data.preprocessing.augmentation.augmentors.factories.instance_augmentor_factory import InstanceAugmentorFactory
from src.data.preprocessing.augmentation.augmentors.photometric.factories.brightness_filter_factory import \
    BrightnessFilterFactory
from src.data.preprocessing.augmentation.augmentors.photometric.factories.color_jitter_filter_factory import \
    ColorJitterFilterFactory
from src.data.preprocessing.augmentation.augmentors.photometric.factories.constrast_filter_factory import \
    ContrastFilterFactory
from src.data.preprocessing.augmentation.augmentors.photometric.factories.gaussian_noise_filter_factory import \
    GaussianNoiseFilterFactory
from src.data.preprocessing.class_balancer_factory import ClassBalancerFactory
from src.data.preprocessing.cond_multiplier_component_factory import CondMultiplierComponentFactory
from src.data.preprocessing.normalization.factories.bbox_normalizer_component_factory import \
    BBoxNormalizerComponentFactory
from src.data.preprocessing.normalization.factories.simple_bbox_normalizer_factory import SimpleBBoxNormalizerFactory
from src.data.preprocessing.resizing.factories.frame_resizer_component_factory import FrameResizerComponentFactory
from src.data.preprocessing.resizing.factories.static_frame_resizer_factory import StaticFrameResizerFactory
from src.data.structures.atomic_bool import AtomicBool
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

@pytest.fixture
def balancer_factory():
    """Fixture to provide a CondMultiplierComponentFactory instance."""
    return ClassBalancerFactory(
        class_counts={
            NorsvinBehaviorClass.BELLY_NOSING: 1885,
            NorsvinBehaviorClass.TAIL_BITING: 1073,
            NorsvinBehaviorClass.EAR_BITING: 1008,
            NorsvinBehaviorClass.TAIL_DOWN: 1107
        },
        max_samples_per=3
    )

def is_annotated(instance: AnnotatedFrame) -> bool:
    """Predicate for checking whether the instance is annotated."""
    return len(instance.annotations) > 0

@pytest.fixture
def multiplier_factory():
    """Fixture to provide a CondMultiplierComponentFactory instance."""
    return CondMultiplierComponentFactory(
        n=2,
        predicate=is_annotated
    )

@pytest.fixture
def augmentor_factory():
    """Fixture to provide a AugmentorComponentFactory instance."""
    augmentor_factory = InstanceAugmentorFactory(
        filter_factories=[
            BrightnessFilterFactory(beta_range=(-30, 30)),
            ContrastFilterFactory(alpha_range=(0.8, 1.2)),
            ColorJitterFilterFactory(saturation_range=(0.8, 1.2), hue_range=(-10, 10)),
            GaussianNoiseFilterFactory(std_range=(5, 15))
        ]
    )

    return AugmentorComponentFactory(
        augmentor_factory=augmentor_factory
    )

import tracemalloc
import threading

tracemalloc.start()


def report_memory():
    current, peak = tracemalloc.get_traced_memory()
    print(f"[Memory] Current = {current / 1024 ** 2:.2f} MB; Peak = {peak / 1024 ** 2:.2f} MB")
    print(f"[Threads] Active threads: {threading.active_count()}")


def test_norsvin_train_stream(gcs_creds, split_ratios, resizer_component_factory, normalizer_component_factory,
                              balancer_factory, multiplier_factory, augmentor_factory):
    """Tests that creating the Norsvin training set stream with GCSStreamFactory gives the correct stream."""
    # arrange
    stream_factory = GCSStreamFactory(
        gcs_creds=gcs_creds,
        split_ratios=split_ratios,
        split=DatasetSplit.TRAIN,
        label_map=NorsvinBehaviorClass.get_label_map(),
        preprocessor_factories=[
            resizer_component_factory,
            normalizer_component_factory,
            balancer_factory,
            multiplier_factory,
            augmentor_factory
        ]
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
