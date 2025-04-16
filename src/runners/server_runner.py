import gc
import time

import numpy as np
from pympler import muppy, summary
import objgraph

from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.managed.factories.norsvin_stream_factory import NorsvinStreamFactory
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
from src.network.messages.requests.handlers.registry.factories.default_handler_registry_factory import \
    DefaultHandlerRegistryFactory
from src.network.messages.serialization.factories.pickle_deserializer_factory import PickleDeserializerFactory
from src.network.messages.serialization.factories.pickle_serializer_factory import PickleSerializerFactory
from src.network.server.network_server import NetworkServer
from src.network.server.session.factories.clean_session_factory import CleanSessionFactory
from src.utils.gcs_credentials import GCSCredentials
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from src.utils.norsvin_dataset_config import NORSVIN_SPLIT_RATIOS
from tests.utils.gcs.test_bucket import TestBucket

import tracemalloc
import threading


def report_memory():
    current, peak = tracemalloc.get_traced_memory()
    print(f"[Memory] Current = {current / 1024 ** 2:.2f} MB; Peak = {peak / 1024 ** 2:.2f} MB")
    print(f"[Threads] Active threads: {threading.active_count()}")


def report_objects():
    all_objects = muppy.get_objects()
    sum1 = summary.summarize(all_objects)
    summary.print_(sum1)


def is_annotated(instance: StreamedAnnotatedFrame) -> bool:
    """Predicate for checking whether the instance is annotated."""
    return len(instance.annotations) > 0


def main():
    gcs_creds = GCSCredentials(bucket_name=TestBucket.BUCKET_NAME, service_account_path=TestBucket.SERVICE_ACCOUNT_FILE)
    split_ratios = NORSVIN_SPLIT_RATIOS

    resizer_factory = FrameResizerComponentFactory(StaticFrameResizerFactory((640, 640)))
    normalizer_factory = BBoxNormalizerComponentFactory(SimpleBBoxNormalizerFactory((0, 1)))
    balancer_factory = ClassBalancerFactory(
        class_counts={
            NorsvinBehaviorClass.BELLY_NOSING: 1885,
            NorsvinBehaviorClass.TAIL_BITING: 1073,
            NorsvinBehaviorClass.EAR_BITING: 1008,
            NorsvinBehaviorClass.TAIL_DOWN: 1107
        },
        max_samples_per=3
    )
    multiplier_factory = CondMultiplierComponentFactory(
        n=2,
        predicate=is_annotated
    )
    augmentor_factory = InstanceAugmentorFactory(
        filter_factories=[
            BrightnessFilterFactory(beta_range=(-30, 30)),
            ContrastFilterFactory(alpha_range=(0.8, 1.2)),
            ColorJitterFilterFactory(saturation_range=(0.8, 1.2), hue_range=(-10, 10)),
            GaussianNoiseFilterFactory(std_range=(5, 15))
        ]
    )
    augmentor_comp_factory = AugmentorComponentFactory(
        augmentor_factory=augmentor_factory,
    )

    train_factories = [
        resizer_factory, normalizer_factory, balancer_factory, multiplier_factory, augmentor_comp_factory
    ]
    val_factories = [
        resizer_factory, normalizer_factory
    ]
    test_factories = [
        resizer_factory, normalizer_factory
    ]

    stream_factory = NorsvinStreamFactory(
        gcs_creds=gcs_creds,
        split_ratios=split_ratios,
        preprocessor_factories=(train_factories, val_factories, test_factories)
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
    stream = stream_factory.create_stream(DatasetSplit.TRAIN)
    stream.run()

    running = AtomicBool(False)

    def log_mem():
        while running:
            report_objects()
            report_memory()
            time.sleep(2)

            arrays = [obj for obj in gc.get_objects() if isinstance(obj, np.ndarray)]
            if arrays:
                biggest = max(arrays, key=lambda a: a.nbytes if hasattr(a, 'nbytes') else 0)
                print(f"Largest array: {biggest.shape}, {biggest.nbytes / 1024:.1f} KB")
                objgraph.show_backrefs([biggest], max_depth=5, filename='leak.png')

    t = threading.Thread(target=log_mem)

    try:
        tracemalloc.start(25)
        running.set(True)
        t.start()
        while True:
            # instance = stream.read()
            time.sleep(.5)
            # report_memory()
            # report_objects()
    except KeyboardInterrupt:
        server.stop()
        t.join()


if __name__ == "__main__":
    main()
