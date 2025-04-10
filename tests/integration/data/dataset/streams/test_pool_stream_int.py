import os

import cv2
import numpy as np
import pytest
import torch
from torchvision.transforms.functional import to_pil_image

from src.auth.factories.gcp_auth_service_factory import GCPAuthServiceFactory
from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.dataset.factories.lazy_entity_factory import LazyEntityFactory
from src.data.dataset.manifests.matching_manifest import MatchingManifest
from src.data.dataset.providers.manifest_instance_provider import ManifestInstanceProvider
from src.data.dataset.registries.suffix_file_registry import SuffixFileRegistry
from src.data.dataset.selectors.random_string_selector import RandomStringSelector
from src.data.dataset.splitters.determ_splitter import DetermSplitter
from src.data.dataset.streams.pool_stream import PoolStream
from src.data.decoders.factories.darwin_decoder_factory import DarwinDecoderFactory
from src.data.label.factories.simple_label_parser_factory import SimpleLabelParserFactory
from src.data.loading.factories.gcs_loader_factory import GCSLoaderFactory
from src.data.parsing.base_name_parser import BaseNameParser
from src.data.preprocessing.augmentation.augmentors.annotated_frame_augmentor import AnnotatedFrameAugmentor
from src.data.preprocessing.augmentation.augmentors.cond_augmentor import CondMultiplier
from src.data.preprocessing.normalization.factories.simple_bbox_normalizer_factory import SimpleBBoxNormalizerFactory
from src.data.preprocessing.resizing.factories.static_frame_resizer_factory import StaticFrameResizerFactory
from src.data.streaming.factories.aggregated_streamer_factory import AggregatedStreamerFactory
from src.data.streaming.factories.file_streamer_pair_factory import FileStreamerPairFactory
from src.data.streaming.managers.static_streamer_manager import StaticStreamerManager
from src.models.converters.xi.ultralytics_batch_converter import UltralyticsBatchConverter
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from tests.utils.gcs.test_bucket import TestBucket
from tests.utils.streamed_annotated_frame_visualizer import StreamedAnnotatedFrameVisualizer


@pytest.fixture
def auth_factory():
    """Fixture to provide an AuthServiceFactory instance."""
    return GCPAuthServiceFactory(TestBucket.SERVICE_ACCOUNT_FILE)


@pytest.fixture
def decoder_factory():
    """Fixture to provide a DecoderFactory instance."""
    return DarwinDecoderFactory(SimpleLabelParserFactory(NorsvinBehaviorClass.get_label_map()))


@pytest.fixture
def loader_factory(auth_factory, decoder_factory):
    """Fixture to provide a LoaderFactory instance."""
    return GCSLoaderFactory(
        bucket_name=TestBucket.BUCKET_NAME,
        auth_factory=auth_factory,
        decoder_factory=decoder_factory
    )


@pytest.fixture
def manifest(loader_factory):
    """Fixture to provide a Manifest instance."""
    source = loader_factory.create_file_registry()
    return MatchingManifest(
        video_registry=SuffixFileRegistry(source=source, suffixes=("mp4",)),
        annotations_registry=SuffixFileRegistry(source=source, suffixes=("json",))
    )


@pytest.fixture
def splitter(manifest):
    """Fixture to provide a DetermSplitter instance."""
    return DetermSplitter(strings=manifest.ids, weights=[0.8, 0.1, 0.1])


@pytest.fixture
def instance_provider(manifest, splitter):
    """Fixture to provide an InstanceProvider instance."""
    return ManifestInstanceProvider(
        manifest=manifest,
        selector=RandomStringSelector(strings=splitter.splits[0])
    )


@pytest.fixture
def entity_factory(loader_factory):
    """Fixture to provide an EntityFactory instance."""
    return LazyEntityFactory(loader_factory, BaseNameParser())


@pytest.fixture
def streamer_pair_factory(instance_provider, entity_factory):
    """Fixture to provide a StreamerPairFactory instance."""
    return FileStreamerPairFactory(
        instance_provider=instance_provider,
        entity_factory=entity_factory,
        frame_resizer_factory=StaticFrameResizerFactory((1920, 1080)),
        bbox_normalizer_factory=SimpleBBoxNormalizerFactory((0, 1))
    )


@pytest.fixture
def streamer_factory(streamer_pair_factory):
    """Fixture to provide an AggregatedStreamerFactory instance."""
    return AggregatedStreamerFactory(streamer_pair_factory)


@pytest.fixture
def multiplier():
    """Fixture to provide a CondMultiplier instance."""

    def is_annotated(instance: StreamedAnnotatedFrame):
        return len(instance.annotations) > 0

    return CondMultiplier(50, is_annotated)


@pytest.fixture
def augmentor():
    """Fixture to provide an Augmentor instance."""
    return AnnotatedFrameAugmentor()


@pytest.fixture
def stream(multiplier, augmentor):
    """Fixture to provide a PoolStream instance."""
    return PoolStream[StreamedAnnotatedFrame](
        pool_size=1000,
        preprocessors=[multiplier, augmentor]
    )


@pytest.fixture
def manager(streamer_factory, stream):
    """Fixture to provide a StaticStreamerManager instance."""
    return StaticStreamerManager[StreamedAnnotatedFrame](
        streamer_factory=streamer_factory,
        consumer=stream
    )


def test_streaming_train_set(stream, manager):
    """Tests that the train set stream gives random training instances."""
    # arrange
    manager.run()

    # act
    instance = stream.read()
    i = 0
    while instance and i < 10000:
        assert isinstance(instance, StreamedAnnotatedFrame)
        instance = stream.read()
        StreamedAnnotatedFrameVisualizer.visualize(instance)
        i += 1
    print(f"Finished reading!")
    manager.stop()



def test_converted_batches_after_streaming(stream, manager):
    """Tests the format of converted batches from streamed AnnotatedFrames (Ultralytics format), and saves 20 annotated images."""
    manager.run()

    save_dir = "/mnt/c/Users/chris/Pictures"
    os.makedirs(save_dir, exist_ok=True)
    saved_count = 0
    max_images = 20

    try:
        for i in range(10):  # Pull up to 10 batches
            frames = []
            for _ in range(4):  # Simulate batch of 4
                frame = stream.read()
                if frame is None:
                    break
                frames.append(frame)

            # Skip empty batch or batch with no annotations
            if not frames or all(len(f.annotations) == 0 for f in frames):
                print(f"[Batch {i}] Skipping due to no annotated frames.")
                continue

            converted = UltralyticsBatchConverter.convert(frames)

            for j, sample in enumerate(converted):
                if saved_count >= max_images:
                    break

                img_tensor = sample["img"]
                img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()

                height, width = img_np.shape[:2]
                bboxes = sample["instances"]["bboxes"]
                classes = sample["instances"]["cls"]

                print(f"[Image {saved_count}] Detected {len(bboxes)} bboxes, classes: {classes.tolist()}")

                for k in range(len(bboxes)):
                    cx, cy, bw, bh, _ = bboxes[k].tolist()
                    x_min = int((cx - bw / 2) * width)
                    y_min = int((cy - bh / 2) * height)
                    x_max = int((cx + bw / 2) * width)
                    y_max = int((cy + bh / 2) * height)

                    class_id = classes[k].item()

                    cv2.rectangle(img_np, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                    cv2.putText(
                        img_np, f"cls {class_id}", (x_min, max(10, y_min - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                    )

                filename = f"converted_sample_{saved_count}.jpg"
                full_path = os.path.join(save_dir, filename)
                cv2.imwrite(full_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                print(f"Saved annotated image to: {full_path}")
                saved_count += 1

            if saved_count >= max_images:
                break

    finally:
        manager.stop()