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
from src.data.providers.instance_provider import InstanceProvider
from src.schemas.observer.schema_broker import SchemaBroker
from src.schemas.pressure_schema import PressureSchema

I = TypeVar("I", bound=Identifiable)
O = TypeVar("O")


class VirtualDataset(Generic[I, O], InstanceProvider[O]):
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

            self._send_pressure_schema(0, batch_size)

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
            occupied=len(self) / self._max_size,
            timestamp=time.time()
        )

    def feed(self, food: I) -> None:
        """
        Feeds an annotated frame.

        Args:
            food (I): the data instance to feed
        """
        id_ = food.get_id()

        with self._lock:
            split_buffer = self._get_split_buffer_for_source(id_)
            evicted = split_buffer.add(*self._get_identified_instance(food))

            # remove evicted instances from the splitter
            for id_ in evicted:
                self._splitter.remove_instance(id_)

            self._send_pressure_schema(1, 0)

    @abstractmethod
    def _get_identified_instance(self, food: I) -> Tuple[str, O]:
        """
        Returns the instance with an identifier.

        Args:
            food (I): the fed data

        Returns:
            Tuple[str, T]: a tuple (str, T) containing the ID and instance
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
