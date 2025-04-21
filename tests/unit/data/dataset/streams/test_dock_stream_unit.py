import threading

import pytest
import time

from src.data.dataset.streams.dock_stream import DockStream
from src.data.structures.atomic_bool import AtomicBool
from src.data.structures.atomic_var import AtomicVar


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
    stream = DockStream[str](buffer_size=2, dock_size=10)

    # act
    entry1 = stream.get_consumer()
    for s in data1:
        entry1.consume(s)

    entry2 = stream.get_consumer()
    for s in data2:
        entry2.consume(s)

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

    entry = stream.get_consumer()
    for s in data1:
        entry.consume(s)

    # act
    stream.close()

    # assert
    for i in range(len(data1) - 1):
        assert stream.read() is not None

    assert stream.read() is None


@pytest.mark.unit
def test_release_unblocks_outer_queue():
    """Tests that a release unblocks the outer queue successfully."""
    # arrange
    stream = DockStream[str](buffer_size=1, dock_size=10)
    stream.get_consumer()

    release = AtomicBool(False)

    span = AtomicVar[float](0.0)

    def put() -> None:
        start_time = time.time()
        stream.get_consumer(release=release)
        end_time = time.time()

        spanned = end_time - start_time
        span.set(spanned)

    t = threading.Thread(target=put)

    sleep_time = 0.2

    # act
    t.start()
    time.sleep(sleep_time)
    release.set(True)
    t.join()

    # assert
    assert span.get() >= sleep_time


@pytest.mark.unit
def test_release_unblocks_inner_queue(data1):
    """Tests that a release unblocks the inner queue successfully."""
    # arrange
    stream = DockStream[str](buffer_size=1, dock_size=1)
    release = AtomicBool(False)
    entry = stream.get_consumer(release=release)
    entry.consume("first-string")

    span = AtomicVar[float](0.0)

    def put() -> None:
        start_time = time.time()
        entry.consume("second-string")
        end_time = time.time()

        spanned = end_time - start_time
        span.set(spanned)

    t = threading.Thread(target=put)

    sleep_time = 0.2

    # act
    t.start()
    time.sleep(sleep_time)
    release.set(True)
    t.join()

    # assert
    assert span.get() >= sleep_time
