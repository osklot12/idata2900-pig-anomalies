from unittest.mock import Mock

import numpy as np
import pytest

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.bbox import BBox
from src.data.dataclasses.frame import Frame
from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.dataclasses.source_metadata import SourceMetadata
from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.parsing.base_name_parser import BaseNameParser
from src.data.streaming.aggregators.buffered_aggregator import BufferedAggregator
from tests.utils.dummy_annotation_label import DummyAnnotationLabel


@pytest.fixture
def matching_source_id():
    """Fixture to provide the source ID for the matching data."""
    return "id1"


@pytest.fixture
def matching_frame(matching_source_id):
    """Fixture to provide a Frame instance that has matching annotations."""
    return Frame(
        source=SourceMetadata("dataset/videos/" + matching_source_id + ".mp4", (200, 100)),
        index=34,
        data=np.random.randint(0, 256, size=(100, 200), dtype=np.uint8)
    )


@pytest.fixture
def matching_annotations(matching_source_id):
    """Fixture to provide a FrameAnnotations instance that has a matching frame."""
    return FrameAnnotations(
        source=SourceMetadata("dataset/annotations/" + matching_source_id + ".json", (200, 100)),
        index=34,
        annotations=[
            AnnotatedBBox(
                cls=DummyAnnotationLabel.CODING,
                bbox=BBox(0.743, 0.3491, 0.12, 0.14325)
            )
        ]
    )


@pytest.fixture
def single_annotations():
    """Fixture to provide a FrameAnnotations instance that has no matching frame."""
    return FrameAnnotations(
        source=SourceMetadata("dataset/annotations/id1.json", (200, 100)),
        index=35,
        annotations=[
            AnnotatedBBox(
                cls=DummyAnnotationLabel.CODING,
                bbox=BBox(0.743, 0.3491, 0.12, 0.14325)
            )
        ]
    )


@pytest.fixture
def consumer():
    """Fixture to provide a mock callback."""
    return Mock()


@pytest.fixture
def aggregator(consumer):
    """Fixture to provide a BufferedInstanceAggregator instance."""
    return BufferedAggregator(consumer)


def _validate_fed_pair(consumer: Mock, frame: Frame, annotations: FrameAnnotations):
    """Validates that the fed StreamedAnnotatedFrame is consistent with the fed frame and annotations."""
    fed_instance = consumer.feed.call_args[0][0]
    assert isinstance(fed_instance, StreamedAnnotatedFrame)

    assert fed_instance.index == frame.index
    assert fed_instance.index == annotations.index

    assert np.array_equal(fed_instance.frame, frame.data)

    assert fed_instance.annotations == annotations.annotations


@pytest.mark.unit
def test_matches_matching_pair_frame_first(aggregator, matching_annotations, matching_frame, consumer):
    """Tests that BufferedInstanceAggregator pairs matching data when frame is fed first."""
    # act
    aggregator.feed_frame(matching_frame)
    aggregator.feed_annotations(matching_annotations)

    # assert
    assert consumer.feed.call_count == 1

    _validate_fed_pair(consumer, matching_frame, matching_annotations)


@pytest.mark.unit
def test_matches_matching_pair_annotations_first(aggregator, matching_annotations, matching_frame, consumer):
    """Tests that BufferedInstanceAggregator pairs matching data even if annotations are fed first."""
    # act
    aggregator.feed_annotations(matching_annotations)
    aggregator.feed_frame(matching_frame)

    # assert
    assert consumer.feed.call_count == 1
    _validate_fed_pair(consumer, matching_frame, matching_annotations)


@pytest.mark.unit
def test_non_matching_data_is_not_matched(aggregator, matching_frame, single_annotations, consumer):
    """Tests that feeding non-matching data to BufferedInstanceAggregator will not cause any matching."""
    # act
    aggregator.feed_frame(matching_frame)
    aggregator.feed_annotations(single_annotations)

    # assert
    assert not consumer.feed.called


@pytest.mark.unit
def test_feeding_non_matching_data_between_matches_still_causes_matching(aggregator, matching_frame,
                                                                         matching_annotations, single_annotations,
                                                                         consumer):
    """Tests that feeding non-matching data to BufferedInstanceAggregator in between matches will cause matching."""
    # act
    aggregator.feed_frame(matching_frame)
    aggregator.feed_annotations(single_annotations)
    aggregator.feed_annotations(matching_annotations)

    # assert
    assert consumer.feed.call_count == 1

    _validate_fed_pair(consumer, matching_frame, matching_annotations)


@pytest.mark.unit
def test_feeding_over_capacity_evicts_stored_data(matching_frame, matching_annotations, single_annotations, consumer):
    """Tests that feeding more than the capacity can hold results in data being evicted."""
    # arrange
    buffer_aggregator = BufferedAggregator(consumer=consumer, buffer_size=1)

    # act
    buffer_aggregator.feed_annotations(matching_annotations)
    buffer_aggregator.feed_annotations(single_annotations)
    buffer_aggregator.feed_frame(matching_frame)

    # assert
    assert not consumer.feed.called


@pytest.mark.unit
def test_feeding_none_frame_and_annotations_signals_end_of_stream(aggregator, consumer):
    """Tests that feeding the BufferedAggregator a None frame and annotations will signal end of stream."""
    # act
    aggregator.feed_frame(None)
    aggregator.feed_annotations(None)

    # assert
    assert consumer.feed.call_count == 1
    assert consumer.feed.call_args[0][0] is None
