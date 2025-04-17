import time
import zlib

import numpy as np
from pympler import muppy, summary

import os
import psutil

from src.data.dataset.streams.managed.factories.gcs_stream_factory import GCSStreamFactory
from src.data.pipeline.factories.norsvin_train_pipeline_factory import NorsvinTrainPipelineFactory
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from tests.utils.streamed_annotated_frame_visualizer import StreamedAnnotatedFrameVisualizer

proc = psutil.Process(os.getpid())

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataset.dataset_split import DatasetSplit
from src.data.structures.atomic_bool import AtomicBool
from src.utils.gcs_credentials import GCSCredentials
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


def is_annotated(instance: AnnotatedFrame) -> bool:
    """Predicate for checking whether the instance is annotated."""
    return len(instance.annotations) > 0


def main():
    gcs_creds = GCSCredentials(bucket_name=TestBucket.BUCKET_NAME, service_account_path=TestBucket.SERVICE_ACCOUNT_FILE)
    split_ratios = NORSVIN_SPLIT_RATIOS

    stream_factory = GCSStreamFactory(
        gcs_creds=gcs_creds,
        split_ratios=split_ratios,
        split=DatasetSplit.TRAIN,
        label_map=NorsvinBehaviorClass.get_label_map(),
        pipeline_factory=NorsvinTrainPipelineFactory()
    )


    # session_factory = CleanSessionFactory()
    # handler_factory = DefaultHandlerRegistryFactory(stream_factory=stream_factory)

    # server = NetworkServer(
    #     serializer_factory=PickleSerializerFactory(),
    #     deserializer_factory=PickleDeserializerFactory(),
    #     session_factory=session_factory,
    #     handler_factory=handler_factory
    # )

    # server.run()
    stream = stream_factory.create_stream()
    stream.run()

    running = AtomicBool(False)

    def log_mem():
        while running:
            report_objects()
            report_memory()
            time.sleep(2)

            mem = proc.memory_info()
            print(f"[System] RSS={mem.rss / 1024 ** 2:.2f} MB | VMS={mem.vms / 1024 ** 2:.2f} MB")

    t = threading.Thread(target=log_mem)

    frames = 0
    last_time = time.time()
    try:
        tracemalloc.start(25)
        running.set(True)
        t.start()
        while True:
            data = stream.read()
            array = np.frombuffer(zlib.decompress(data.frame), dtype=np.dtype(data.dtype)).reshape(data.shape)
            writable_array = np.copy(array)  # Now it's a writable copy

            frame = AnnotatedFrame(
                source=data.source,
                index=data.index,
                frame=writable_array,
                annotations=data.annotations
            )

            frames += 1
            if frames > 10:
                print(f"FPS: {frames / (time.time() - last_time)}")
                last_time = time.time()
                frames = 0

            StreamedAnnotatedFrameVisualizer.visualize(frame)
            # report_memory()
            # report_objects()
    except KeyboardInterrupt:
        # server.stop()
        t.join()


if __name__ == "__main__":
    main()
