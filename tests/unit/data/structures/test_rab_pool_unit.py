import queue
import threading

import pytest
import time

from src.data.structures.atomic_bool import AtomicBool
from src.data.structures.atomic_var import AtomicVar
from src.data.structures.rab_pool import RABPool


@pytest.fixture
def data():
    """Fixture to provide test data."""
    return [f"string{i}" for i in range(1000)]


@pytest.mark.unit
def test_get_returns_none_when_empty(data):
    """Tests that get() returns None when pool is empty."""
    # arrange
    pool = RABPool[str](maxsize=1000, min_ready=0)

    # act
    instance = pool.get()

    # assert
    assert instance is None


@pytest.mark.unit
def test_get_blocks_when_min_ready_is_not_met(data):
    """Tests that the pool blocks on get() if there are not at least min_ready instances in the pool."""
    # arrange
    pool = RABPool[str](maxsize=1001, min_ready=1001)
    for s in data:
        pool.put(s)

    release = AtomicBool(False)

    span = AtomicVar[float](0.0)

    def put() -> None:
        start_time = time.time()
        pool.get(release=release)
        end_time = time.time()
        span.set(end_time - start_time)

    t = threading.Thread(target=put)
    sleep_time = 0.1

    # act
    t.start()
    time.sleep(sleep_time)
    release.set(True)
    t.join()

    # assert
    assert span.get() >= sleep_time


@pytest.mark.unit
def test_get_does_not_block_when_min_ready_is_met(data):
    """Tests that the pool does not block on get() if there are at least min_ready instances in the pool."""
    # arrange
    pool = RABPool[str](maxsize=1000, min_ready=999)
    for s in data:
        pool.put(s)

    release = AtomicBool(False)

    span = AtomicVar[float](0.0)

    item = AtomicVar[str]("")

    def get() -> None:
        start_time = time.time()
        item.set(pool.get(release=release))
        end_time = time.time()
        span.set(end_time - start_time)

    t = threading.Thread(target=get)
    sleep_time = 0.1

    # act
    t.start()
    time.sleep(sleep_time)
    release.set(True)
    t.join()

    # assert
    assert span.get() < sleep_time
    assert item.get() is not ""


@pytest.mark.unit
def test_put_blocks_on_full_pool(data):
    """Tests that the put() blocks if the pool is full."""
    # arrange
    pool = RABPool[str](maxsize=1000, min_ready=999)
    for s in data:
        pool.put(s)

    release = AtomicBool(False)

    span = AtomicVar[float](0.0)

    def get() -> None:
        start_time = time.time()
        pool.put(item="block", release=release)
        end_time = time.time()
        span.set(end_time - start_time)

    t = threading.Thread(target=get)
    sleep_time = 0.1

    # act
    t.start()
    time.sleep(sleep_time)
    release.set(True)
    t.join()

    # assert
    assert span.get() >= sleep_time


@pytest.mark.unit
def test_get_removes_instance(data):
    """Tests that get() removes the returned instance from the pool."""
    # arrange
    pool = RABPool[str](maxsize=1000, min_ready=0)
    for s in data:
        pool.put(s)
    instances = []

    # act
    for _ in range(len(data)):
        instance = pool.get()
        instances.append(instance)

    last_instance = pool.get()

    # assert
    assert len(instances) == len(data)
    assert last_instance is None


@pytest.mark.unit
def test_get_returns_random_instance(data):
    """Tests that get() returns random instances in different orders."""
    # arrange
    n_trials = 10
    pools = [RABPool[str](maxsize=1000, min_ready=0) for _ in range(n_trials)]
    for pool in pools:
        for s in data:
            pool.put(s)
    all_sequences = []

    # act
    for pool in pools:
        instances = []
        for _ in range(len(data)):
            instances.append(pool.get())
        all_sequences.append(instances)

    # assert
    unique_sequences = set(tuple(seq) for seq in all_sequences)
    assert len(unique_sequences) > 1


@pytest.mark.unit
def test_thread_safety(data):
    """Tests that put() and get() are thread-safe."""
    # arrange
    pool = RABPool[str](maxsize=1000, min_ready=0)
    instances = queue.Queue()

    def put_worker(start, end):
        for i in range(start, end):
            pool.put(data[i])

    def get_worker(n):
        for i in range(n):
            instances.put(pool.get())

    n_threads = 10
    put_threads = [
        threading.Thread(target=put_worker(int((len(data) / n_threads) * i), int((len(data) / n_threads) * (i + 1))))
        for i in range(n_threads)
    ]
    get_threads = [
        threading.Thread(target=get_worker(int(len(data) / n_threads))) for i in range(n_threads)
    ]

    # act
    for t in put_threads:
        t.start()

    for t in get_threads:
        t.start()

    for t in put_threads:
        t.join()

    for t in get_threads:
        t.join()

    # assert
    assert instances.qsize() == len(data)
    assert sorted(list(instances.queue)) == sorted(data)
