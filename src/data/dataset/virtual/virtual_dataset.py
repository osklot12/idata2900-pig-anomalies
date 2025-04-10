import math
import threading
import random
import time
from abc import abstractmethod
from typing import List, TypeVar, Generic, Tuple

from src.data.structures.hash_buffer import HashBuffer
from src.data.dataclasses.identifiable import Identifiable
from src.data.dataset.splitters.dataset_splitter import DatasetSplitter
from src.data.dataset.dataset_split import DatasetSplit
from src.data.providers.batch_provider import BatchProvider
from src.schemas.brokers.schema_broker import SchemaBroker
from src.schemas.schemas.pressure_schema import PressureSchema

I = TypeVar("I", bound=Identifiable)
O = TypeVar("O")


class VirtualDataset(Generic[I, O], BatchProvider[O]):
    """A thread-safe, split-aware buffer system for managing annotated video frames in memory."""

    def __init__(self, splitter: DatasetSplitter, max_size: int,
                 pressure_broker: SchemaBroker[PressureSchema] = None):
        """
        Initializes a VirtualDataset instance.

        Args:
            splitter (DatasetSplitter): object responsible for managing dataset splits
            max_size (int): the maximum number of simultaneous instances stored
            pressure_broker (SchemaBroker[PressureSchema]): broker for notifying about pressure
        """
        self._splitter = splitter

        self._lock = threading.Lock()

        self._max_size = max_size

        # floor each max buffer size to never exceed the total max buffer size
        train_max_sources = math.floor(self._splitter.get_split_ratio(DatasetSplit.TRAIN) * max_size)
        val_max_sources = math.floor(self._splitter.get_split_ratio(DatasetSplit.VAL) * max_size)
        test_max_sources = math.floor(self._splitter.get_split_ratio(DatasetSplit.TEST) * max_size)

        # split buffers
        self._train_buffer = HashBuffer[str, O](max_size=train_max_sources)
        self._val_buffer = HashBuffer[str, O](max_size=val_max_sources)
        self._test_buffer = HashBuffer[str, O](max_size=test_max_sources)

        self._buffer_map = {
            DatasetSplit.TRAIN: self._train_buffer,
            DatasetSplit.VAL: self._val_buffer,
            DatasetSplit.TEST: self._test_buffer
        }

        self._event_broker = pressure_broker

    def get_batch(self, split: DatasetSplit, batch_size: int) -> List[O]:
        with self._lock:
            buffer = self._buffer_map.get(split)
            if len(buffer) < batch_size:
                raise ValueError(f"Not enough available data in split {split.name} for batch size {batch_size}.")

            return random.sample(buffer, batch_size)

    def _send_pressure_schema(self, inputs: int, outputs: int):
        """Sends a pressure schema."""
        if self._event_broker:
            self._event_broker.notify(self._create_pressure_schema(inputs, outputs))

    def _create_pressure_schema(self, inputs: int, outputs: int) -> PressureSchema:
        """Creates a pressure schema."""
        return PressureSchema(
            inputs=inputs,
            outputs=outputs,
            usage=len(self) / self._max_size,
            timestamp=time.time()
        )

    def feed(self, food: I) -> None:
        """
        Feeds an annotated frame.

        Args:
            food (I): the data instance to feed
        """
        with self._lock:
            instance_id = self._get_instance_id(food)
            split_buffer = self._get_split_buffer_for_source(instance_id)
            evicted = split_buffer.add(instance_id, food)

            # remove evicted instances from the splitter
            for id_ in evicted:
                self._splitter.remove_instance(id_)

    @abstractmethod
    def _get_instance_id(self, food: I) -> str:
        """
        Returns the instance ID.

        Args:
            food (I): the instance to get ID for

        Returns:
            str: the instance ID
        """
        raise NotImplementedError

    def __len__(self) -> int:
        return sum(len(buf) for buf in self._buffer_map.values())

    @abstractmethod
    def _frame_in_instance(self, instance: O) -> int:
        """
        Returns the number of frames in the instance.

        Args:
            instance (T): the instance to get frame count for

        Returns:
            int: the number of frames in the instance
        """
        raise NotImplementedError

    def split_size(self, split: DatasetSplit) -> int:
        """
        Returns the size of the given split.

        Args:
            split (DatasetSplit): the split to get size of

        Returns:
            int: the size of the given split
        """
        return len(self._buffer_map.get(split))

    def _get_split_buffer_for_source(self, source_id: str) -> HashBuffer:
        """Returns the split buffer for the given source id."""
        split = self._splitter.get_split_for_id(source_id)

        # updating splits if source was not recognized
        if split is None:
            split = self._splitter.add_instance(source_id)

        if split is None:
            raise RuntimeError(f"Splitter failed to assign a split for source: {source_id}")

        buffer = self._buffer_map.get(split)

        return buffer