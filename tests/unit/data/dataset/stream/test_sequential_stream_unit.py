import queue
from typing import List, Optional

import pytest

from src.data.dataset.streams.dock_stream import DockStream


@pytest.fixture
def data1():
    """Fixture to provide test streaming data."""
    return [
        "data1", "data2", "data3", "data4", "data5", "data6", None
    ]


@pytest.fixture
def data2():
    """Fixture to provide test streaming data."""
    return [
        "data7", "data8", "data9", "data10", "data11", "data12", None
    ]


@pytest.mark.unit
def test_input_data_is_read_sequentially(data1, data2):
    """Tests that the input data is read sequentially in the expected order."""
    # arrange
    stream = DockStream[str]()

    # act
    feedable1 = stream.dock()
    for s in data1:
        feedable1.consume(s)

    feedable2 = stream.dock()
    for s in data2:
        feedable2.consume(s)

    # assert
    for i in range(len(data1) - 1):
        assert stream.read() == data1[i]

    for i in range(len(data2) - 1):
        assert stream.read() == data2[i]


@pytest.mark.unit
def test_close_closes_stream(data1):
    """Tests that calling close closes the streams."""
    # arrange
    stream = DockStream[str]()

    feedable = stream.dock()
    for s in data1:
        feedable.consume(s)

    # act
    stream.close()

    # assert
    for i in range(len(data1) - 1):
        assert stream.read() is not None

    assert stream.read() is None
