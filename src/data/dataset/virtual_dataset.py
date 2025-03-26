import math
import random
import threading
from typing import List

from src.data.data_structures.hash_buffer import HashBuffer
from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.dataset.dataset import Dataset
from src.data.dataset.splitters.dataset_splitter import DatasetSplitter
from src.data.dataset_split import DatasetSplit
from src.data.loading.feed_status import FeedStatus


class VirtualDataset(Dataset):
    """A thread-safe, split-aware buffer system for managing annotated video frames in memory."""

    def __init__(self, splitter: DatasetSplitter, max_sources: int, max_frames_per_source: int):
        """
        Initializes a VirtualDataset instance.

        Args:
            splitter (DatasetSplitter): object responsible for managing dataset splits
            max_sources (int): the maximum number of sources
            max_frames_per_source (int): the maximum number of frames per source
        """
        self._splitter = splitter

        self.lock = threading.Lock()

        # floor each max buffer size to never exceed the total max buffer size
        train_max_sources = math.floor(self._splitter.get_split_ratio(DatasetSplit.TRAIN) * max_sources)
        val_max_sources = math.floor(self._splitter.get_split_ratio(DatasetSplit.VAL) * max_sources)
        test_max_sources = math.floor(self._splitter.get_split_ratio(DatasetSplit.TEST) * max_sources)

        # train split buffer
        self._train_buffer = HashBuffer[str, HashBuffer[int, AnnotatedFrame]](max_size=train_max_sources)

        # val split buffer
        self._val_buffer = HashBuffer[str, HashBuffer[int, AnnotatedFrame]](max_size=val_max_sources)

        # test split buffer
        self._test_buffer = HashBuffer[str, HashBuffer[int, AnnotatedFrame]](max_size=test_max_sources)

        self._buffer_map = {
            DatasetSplit.TRAIN: self._train_buffer,
            DatasetSplit.VAL: self._val_buffer,
            DatasetSplit.TEST: self._test_buffer
        }

        self._max_frames_per_source = max_frames_per_source

    def get_shuffled_batch(self, split: DatasetSplit, batch_size: int) -> List[AnnotatedFrame]:
        with self.lock:
            if self.get_frame_count(split) < batch_size:
                raise ValueError(f"Not enough available data in split {split.name} for batch size {batch_size}.")

            buffer = self._buffer_map.get(split)

            frame_references = [
                source_buffer.at(frame_index)
                for source_key in buffer.keys()
                for source_buffer in [buffer.at(source_key)]
                if source_buffer and source_buffer.size() > 0
                for frame_index in source_buffer.keys()
            ]

            # ensure "non-deterministic" behavior
            random.seed(None)

            return random.sample(frame_references, batch_size)

    def feed(self, anno_frame: StreamedAnnotatedFrame) -> FeedStatus:
        """
        Feeds an annotated frame into the virtual dataset.

        Args:
            anno_frame (StreamedAnnotatedFrame): the data instance to feed
        """
        source_id = anno_frame.source.source_id

        with self.lock:
            split_buffer = self._get_split_buffer_for_source(source_id)

            if split_buffer is not None:
                # allocates buffer for source if new
                source_buffer = self._get_source_buffer(source_id, split_buffer)

                # add instance to source buffer
                evicted = source_buffer.add(
                    anno_frame.index,
                    AnnotatedFrame(anno_frame.frame, anno_frame.annotations)
                )

                for id_ in evicted:
                    self._splitter.remove_instance(id_)
                print(f"[VirtualDataset] Received frame {anno_frame.index}")
            else:
                print(f"[VirtualDataset] Source {anno_frame.source} not recognized.")

    def get_frame_count(self, split: DatasetSplit) -> int:
        """
        Returns the total number of stored frames in the specified dataset split.

        Args:
            split (DatasetSplit): the dataset split to get count for
        """
        buffer = self._buffer_map.get(split)
        total_frames = 0

        for source_key in buffer.keys():
            source_buffer = buffer.at(source_key)
            if source_buffer:
                total_frames += source_buffer.size()

        return total_frames

    def _get_split_buffer_for_source(self, source_id: str) -> HashBuffer:
        """Returns the split buffer for the given source id."""
        buffer = None

        split = self._splitter.get_split_for_id(source_id)

        # updating splits if source was not recognized
        if split is None:
            self._splitter.add_instance(source_id)
            split = self._splitter.get_split_for_id(source_id)

        if split is None:
            raise RuntimeError(f"Splitter failed to assign a split for source: {source_id}")

        buffer = self._buffer_map.get(split)

        return buffer

    def _get_source_buffer(self, source_id: str, split_buffer: HashBuffer) -> HashBuffer:
        """Returns the buffer for the given source id, or creates a new one if none exists."""
        if not split_buffer.has(source_id):
            new_buffer = HashBuffer[int, AnnotatedFrame](max_size=self._max_frames_per_source)
            split_buffer.add(source_id, new_buffer)

        return split_buffer.at(source_id)
