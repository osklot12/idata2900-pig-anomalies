from unittest.mock import Mock

import numpy as np
import pytest

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.bbox import BBox
from src.data.dataclasses.frame import Frame
from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.parsing.file_base_name_parser import FileBaseNameParser
from src.data.streaming.aggregators.buffered_instance_aggregator import BufferedInstanceAggregator
from tests.utils.annotation_label import AnnotationLabel


@pytest.fixture
def matching_source_id():
    """Fixture to provide the source ID for the matching data."""
    return "id1"


@pytest.fixture
def matching_frame(matching_source_id):
    """Fixture to provide a Frame instance that has matching annotations."""
    return Frame(
        source="dataset/videos/" + matching_source_id + ".mp4",
        index=34,
        data=np.random.randint(0, 256, size=(100, 200), dtype=np.uint8),
        end_of_stream=False
    )


@pytest.fixture
def matching_annotations(matching_source_id):
    """Fixture to provide a FrameAnnotations instance that has a matching frame."""
    return FrameAnnotations(
        source="dataset/annotations/" + matching_source_id + ".json",
        index=34,
        annotations=[
            AnnotatedBBox(
                cls=AnnotationLabel.CODING,
                bbox=BBox(0.743, 0.3491, 0.12, 0.14325)
            )
        ],
        end_of_stream=False
    )


@pytest.fixture
def single_annotations():
    """Fixture to provide a FrameAnnotations instance that has no matching frame."""
    return FrameAnnotations(
        source="dataset/annotations/id1.json",
        index=35,
        annotations=[
            AnnotatedBBox(
                cls=AnnotationLabel.CODING,
                bbox=BBox(0.743, 0.3491, 0.12, 0.14325)
            )
        ],
        end_of_stream=False
    )


@pytest.fixture
def callback():
    """Fixture to provide a mock callback."""
    return Mock()


@pytest.fixture
def aggregator(callback):
    """Fixture to provide a BufferedInstanceAggregator instance."""
    return BufferedInstanceAggregator(callback)


@pytest.mark.unit
def _validate_fed_pair(callback: Mock, frame: Frame, annotations: FrameAnnotations):
    """Validates that the fed StreamedAnnotatedFrame is consistent with the fed frame and annotations."""
    fed_instance = callback.call_args[0][0]
    assert isinstance(fed_instance, StreamedAnnotatedFrame)

    assert fed_instance.index == frame.index
    assert fed_instance.index == annotations.index

    assert np.array_equal(fed_instance.frame, frame.data)

    assert fed_instance.annotations == annotations.annotations

    assert fed_instance.end_of_stream == frame.end_of_stream
    assert fed_instance.end_of_stream == annotations.end_of_stream


@pytest.mark.unit
def test_matches_matching_pair_frame_first(aggregator, matching_annotations, matching_frame, callback):
    """Tests that BufferedInstanceAggregator pairs matching data when frame is fed first."""
    # act
    aggregator.feed_frame(matching_frame)
    aggregator.feed_annotations(matching_annotations)

    # assert
    assert callback.call_count == 1

    _validate_fed_pair(callback, matching_frame, matching_annotations)


@pytest.mark.unit
def test_matches_matching_pair_annotations_first(aggregator, matching_annotations, matching_frame, callback):
    """Tests that BufferedInstanceAggregator pairs matching data even if annotations are fed first."""
    # act
    aggregator.feed_annotations(matching_annotations)
    aggregator.feed_frame(matching_frame)

    # assert
    assert callback.call_count == 1
    _validate_fed_pair(callback, matching_frame, matching_annotations)


@pytest.mark.unit
def test_non_matching_data_is_not_matched(aggregator, matching_frame, single_annotations, callback):
    """Tests that feeding non-matching data to BufferedInstanceAggregator will not cause any matching."""
    # act
    aggregator.feed_frame(matching_frame)
    aggregator.feed_annotations(single_annotations)

    # assert
    assert not callback.called


@pytest.mark.unit
def test_feeding_non_matching_data_between_matches_still_causes_matching(aggregator, matching_frame,
                                                                         matching_annotations, single_annotations,
                                                                         callback):
    """Tests that feeding non-matching data to BufferedInstanceAggregator in between matches will cause matching."""
    # act
    aggregator.feed_frame(matching_frame)
    aggregator.feed_annotations(single_annotations)
    aggregator.feed_annotations(matching_annotations)

    # assert
    assert callback.call_count == 1

    _validate_fed_pair(callback, matching_frame, matching_annotations)


@pytest.mark.unit
def test_feeding_over_capacity_evicts_stored_data(matching_frame, matching_annotations, single_annotations, callback):
    """Tests that feeding more than the capacity can hold results in data being evicted."""
    # arrange
    buffer_aggregator = BufferedInstanceAggregator(callback=callback, buffer_size=1)

    # act
    buffer_aggregator.feed_annotations(matching_annotations)
    buffer_aggregator.feed_annotations(single_annotations)
    buffer_aggregator.feed_frame(matching_frame)

    # assert
    assert not callback.called


@pytest.mark.unit
def test_feed_none_frame_raises(aggregator):
    """Tests that feeding a frame equal None raises an exception."""
    # act & assert
    with pytest.raises(ValueError, match="frame cannot be None"):
        aggregator.feed_frame(None)


@pytest.mark.unit
def test_feed_none_annotation_raises(aggregator):
    """Tests that feeding annotations equal None raises an exception."""
    # act & assert
    with pytest.raises(ValueError, match="annotations cannot be None"):
        aggregator.feed_annotations(None)
