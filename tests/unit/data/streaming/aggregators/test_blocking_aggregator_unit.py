import threading
import pytest
import time
from unittest.mock import Mock

from src.data.dataclasses.frame import Frame
from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.dataclasses.source_metadata import SourceMetadata
from src.data.streaming.aggregators.blocking_aggregator import BlockingAggregator
from src.data.structures.atomic_bool import AtomicBool
from src.data.structures.atomic_var import AtomicVar


@pytest.fixture
def consumer():
    """Fixture to provide a mock consumer."""
    return Mock()


@pytest.fixture
def aggregator():
    """Fixture to provide a BlockingAggregator instance."""
    return BlockingAggregator()


@pytest.fixture
def release():
    """Fixture to provide a release."""
    return AtomicBool(False)


@pytest.fixture
def source():
    """Fixture to provide a dummy source."""
    return SourceMetadata(source_id="test", frame_resolution=(1920, 1080))


@pytest.mark.unit
def test_feed_frame_blocks_until_match_found(consumer, aggregator, release, source):
    """Tests that feeding a frame to the aggregator will block until a matching annotation is fed."""
    # arrange
    frame = Frame(source=source, index=0, data=None)
    annotations = FrameAnnotations(source=source, index=0, annotations=[])

    timer = AtomicVar[float](0.0)

    def feed_frame() -> None:
        start_time = time.time()
        aggregator.feed_frame(frame)
        end_time = time.time()
        timer.set(end_time - start_time)

    t = threading.Thread(target=feed_frame)
    sleep_time = 0.1

    # act
    t.start()
    time.sleep(sleep_time)
    aggregator.feed_annotations(annotations)
    t.join()

    # assert
    assert timer.get() >= sleep_time


@pytest.mark.unit
def test_feed_anno_blocks_until_match_found(consumer, aggregator, release, source):
    """Tests that feeding annotations to the aggregator will block until a matching frame is fed."""
    # arrange
    frame = Frame(source=source, index=0, data=None)
    annotations = FrameAnnotations(source=source, index=0, annotations=[])

    timer = AtomicVar[float](0.0)

    def feed_anno() -> None:
        start_time = time.time()
        aggregator.feed_annotations(annotations)
        end_time = time.time()
        timer.set(end_time - start_time)

    t = threading.Thread(target=feed_anno)
    sleep_time = 0.1

    # act
    t.start()
    time.sleep(sleep_time)
    aggregator.feed_frame(frame)
    t.join()

    # assert
    assert timer.get() >= sleep_time


@pytest.mark.unit
def test_release_unblocks_feed_frame(consumer, aggregator, release, source):
    """Tests that using a release can unblock the feeding of frames."""
    # arrange
    frame = Frame(source=source, index=0, data=None)

    timer = AtomicVar[float](0.0)
    release = AtomicBool(False)

    def feed_frame() -> None:
        start_time = time.time()
        aggregator.feed_frame(frame=frame, release=release)
        end_time = time.time()
        timer.set(end_time - start_time)

    t = threading.Thread(target=feed_frame)
    sleep_time = 0.1

    # act
    t.start()
    time.sleep(sleep_time)
    release.set(True)
    t.join()

    # assert
    assert timer.get() >= sleep_time


@pytest.mark.unit
def test_release_unblocks_feed_annotations(consumer, aggregator, release, source):
    """Tests that using a release can unblock the feeding of annotations."""
    # arrange
    annotations = FrameAnnotations(source=source, index=0, annotations=[])

    timer = AtomicVar[float](0.0)
    release = AtomicBool(False)

    def feed_frame() -> None:
        start_time = time.time()
        aggregator.feed_annotations(annotations=annotations, release=release)
        end_time = time.time()
        timer.set(end_time - start_time)

    t = threading.Thread(target=feed_frame)
    sleep_time = 0.1

    # act
    t.start()
    time.sleep(sleep_time)
    release.set(True)
    t.join()

    # assert
    assert timer.get() >= sleep_time
