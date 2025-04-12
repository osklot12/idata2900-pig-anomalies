import threading
import time

import pytest

from src.data.dataset.streams.pool_stream import PoolStream
from src.data.structures.atomic_bool import AtomicBool
from src.data.structures.atomic_var import AtomicVar


@pytest.fixture
def data():
    """Fixture to provide test data."""
    return [f"string{i}" for i in range(1000)]


@pytest.mark.unit
def test_feeding_when_not_full(data):
    """Tests that feeding the PoolStream while not full should not block, and feed successfully."""
    # arrange
    stream = PoolStream[str](pool_size=1000)
    successes = []

    # act
    for d in data:
        successes.append(stream.get_entry().consume(d))

    # assert
    assert all(successes)


@pytest.mark.unit
def test_release_mechanism(data):
    """Tests that feeding the PoolStream with a release should unblock when release is set."""
    # arrange
    stream = PoolStream[str](pool_size=1000)
    for d in data:
        stream.get_entry().consume(d)

    release = AtomicBool(False)
    consumer = stream.get_entry(release=release)
    span = AtomicVar[float](0.0)

    def put() -> None:
        start_time = time.time()
        consumer.consume("block-string")
        end_time = time.time()

        new_span = span.get() + end_time - start_time
        span.set(new_span)

    t = threading.Thread(target=put)
    sleep_time = 0.2

    # act
    t.start()
    time.sleep(sleep_time)
    release.set(True)
    t.join()

    # assert
    assert span.get() >= sleep_time
