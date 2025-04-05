import queue
from typing import List, Optional

import pytest

from src.data.dataset.streams.sequential_stream import SequentialStream


@pytest.fixture
def data1():
    """Fixture to provide test streaming data."""
    return [
        "data1", "data2", "data3", "data4", "data5", "data6"
    ]


@pytest.fixture
def data2():
    """Fixture to provide test streaming data."""
    return [
        "data7", "data8", "data9", "data10", "data11", "data12"
    ]


def _create_queue(items: List[str]) -> queue.Queue[Optional[str]]:
    """Creates a queue from a list of strings."""
    q = queue.Queue()
    for item in items:
        q.put(item)
    q.put(None)
    return q


@pytest.mark.unit
def test_stream_is_sequential(data1, data2):
    """Tests that the SequentialStream reads the stream in a sequential manner."""
    # arrange
    stream = SequentialStream[str]()
    stream.queue.put(_create_queue(data1))
    stream.queue.put(_create_queue(data2))
    stream.queue.put(None)
    read_items = []

    # act
    item = stream.read()
    while item is not None:
        read_items.append(item)
        item = stream.read()

    # assert
    for i in range(len(data1)):
        assert read_items[i] == data1[i]

    for i in range(len(data2)):
        assert read_items[i + len(data1)] == data2[i]


def test_consecutive_reads_after_eos_returns_none():
    """Tests that consecutive calls to read returns None after eos."""
    # arrange
    stream = SequentialStream[str]()
    stream.queue.put(None)

