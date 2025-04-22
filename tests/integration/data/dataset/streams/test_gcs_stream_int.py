import numpy as np

from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.selectors.factories.determ_string_selector_factory import DetermStringSelectorFactory
from src.data.dataset.selectors.factories.random_string_selector_factory import RandomStringSelectorFactory
from src.data.dataset.streams.factories.dock_stream_factory import DockStreamFactory
from src.data.dataset.streams.factories.pool_stream_factory import PoolStreamFactory
from src.data.dataset.streams.managed.factories.gcs_stream_factory import GCSStreamFactory
from src.data.pipeline.factories.norsvin_eval_pipeline_factory import NorsvinEvalPipelineFactory
from src.data.pipeline.factories.norsvin_train_pipeline_factory import NorsvinTrainPipelineFactory
from src.data.processing.zlib_decompressor import ZlibDecompressor
from src.models.converters.yolox_batch_converter import YOLOXBatchConverter
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from src.utils.norsvin_dataset_config import NORSVIN_GCS_CREDS, NORSVIN_SPLIT_RATIOS
from tests.utils.annotated_frame_visualizer import AnnotatedFrameVisualizer
from tests.utils.yolox_batch_visualizer import YOLOXBatchVisualizer


def test_norsvin_train_stream():
    """Tests that creating the Norsvin training set stream with GCSStreamFactory gives the correct stream."""
    # arrange
    stream_factory = GCSStreamFactory(
        gcs_creds=NORSVIN_GCS_CREDS,
        split_ratios=NORSVIN_SPLIT_RATIOS,
        split=DatasetSplit.TRAIN,
        selector_factory=RandomStringSelectorFactory(),
        label_map=NorsvinBehaviorClass.get_label_map(),
        stream_factory=PoolStreamFactory(pool_size=3000, min_ready=2000),
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


def test_visualize_yolox_converted_batches():
    """Visualizes the coverted YOLOX batches for manual inspection."""
    # arrange
    stream_factory = GCSStreamFactory(
        gcs_creds=NORSVIN_GCS_CREDS,
        split_ratios=NORSVIN_SPLIT_RATIOS,
        split=DatasetSplit.TRAIN,
        selector_factory=RandomStringSelectorFactory(),
        label_map=NorsvinBehaviorClass.get_label_map(),
        stream_factory=PoolStreamFactory(pool_size=2000, min_ready=300),
        pipeline_factory=NorsvinTrainPipelineFactory()
    )

    stream = stream_factory.create_stream()
    decompressor = ZlibDecompressor()
    converter = YOLOXBatchConverter()
    labels = [
        "TAIL BITING",
        "EAR BITING",
        "BELLY NOSING",
        "TAIL DOWN"
    ]

    batch_size = 8

    # act
    try:
        stream.run()
        while True:
            batch = []
            while len(batch) < batch_size:
                instance = stream.read()
                decompressed = decompressor.process(instance)
                batch.append(decompressed)
            images, targets, _, _ = converter.convert(batch)
            YOLOXBatchVisualizer.visualize(images=images, targets=targets, class_names=labels)
            batch.clear()
    except KeyboardInterrupt:
        stream.stop()


def test_norsvin_val_stream():
    """Tests that creating the Norsvin validation set stream with GCSStreamFactory gives the correct stream."""
    # arrange
    stream_factory = GCSStreamFactory(
        gcs_creds=NORSVIN_GCS_CREDS,
        split_ratios=NORSVIN_SPLIT_RATIOS,
        split=DatasetSplit.VAL,
        selector_factory=DetermStringSelectorFactory(),
        label_map=NorsvinBehaviorClass.get_label_map(),
        stream_factory=DockStreamFactory(buffer_size=6, dock_size=1000),
        pipeline_factory=NorsvinEvalPipelineFactory()
    )

    stream=stream_factory.create_stream()
    stream.run()
    decompressor = ZlibDecompressor()

    # act
    instance = stream.read()
    i = 1
    videos = {}
    while instance:
        AnnotatedFrameVisualizer.visualize(decompressor.process(instance))
        if not instance.source.source_id in videos:
            videos[instance.source.source_id] = 0
        else:
            videos[instance.source.source_id] += 1
        print(f"[Test] Frames read: {i}")
        print(f"[Test] Read frame {instance.index} for {instance.source.source_id}")
        instance = stream.read()

        i += 1
    print(f"[Test] Videos: {videos}")
    stream.stop()


def test_norsvin_test_stream():
    """Tests that creating the Norsvin test set stream with GCSStreamFactory gives the correct stream."""
    # arrange
    stream_factory = GCSStreamFactory(
        gcs_creds=NORSVIN_GCS_CREDS,
        split_ratios=NORSVIN_SPLIT_RATIOS,
        split=DatasetSplit.TEST,
        selector_factory=DetermStringSelectorFactory(),
        label_map=NorsvinBehaviorClass.get_label_map(),
        stream_factory=DockStreamFactory(buffer_size=6, dock_size=1000),
        pipeline_factory=NorsvinEvalPipelineFactory()
    )

    stream = stream_factory.create_stream()
    stream.run()
    decompressor = ZlibDecompressor()

    # act
    instance = stream.read()
    i = 1
    while instance:
        # AnnotatedFrameVisualizer.visualize(decompressor.process(instance))
        print(f"Frames read: {i}")
        print(f"[YOLOXDataset] Read frame {instance.index} for {instance.source.source_id}")
        instance = stream.read()
        i += 1

    stream.stop()