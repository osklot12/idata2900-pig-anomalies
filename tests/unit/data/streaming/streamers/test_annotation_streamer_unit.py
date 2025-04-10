import time
from typing import Optional
from unittest.mock import MagicMock

import pytest

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.bbox import BBox
from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.dataclasses.source_metadata import SourceMetadata
from src.data.preprocessing.normalization.normalizers.bbox_normalizer import BBoxNormalizer
from src.data.preprocessing.normalization.normalizers.simple_bbox_normalizer import SimpleBBoxNormalizer
from src.data.streaming.feedables.feedable import Feedable
from src.data.streaming.streamers.annotation_streamer import AnnotationStreamer
from src.data.streaming.streamers.streamer_status import StreamerStatus

from tests.utils.dummy_annotation_label import DummyAnnotationLabel


class DummyAnnotationStreamer(AnnotationStreamer):
    """A testable implementation of the abstract class AnnotationStreamer."""

    def __init__(self, n_annotations: int, consumer: Feedable[FrameAnnotations],
                 normalizer: BBoxNormalizer = None):
        """
        Initializes a DummyAnnotationStreamer.

        Args:
            n_annotations (int): number of annotations in the streams
            consumer (Feedable[FrameAnnotations]): the consumer of the streaming annotations
            normalizer (BBoxNormalizer): the normalization strategy for normalizing bounding boxes
        """
        super().__init__(consumer, normalizer)
        self.n_annotations = n_annotations
        self.frame_index = 0

    def _get_next_annotation(self) -> Optional[FrameAnnotations]:
        annotation = None

        if self.frame_index < self.n_annotations:
            time.sleep(.005)
            annotations_list = [
                AnnotatedBBox(
                    cls=DummyAnnotationLabel.CODING,
                    bbox=BBox(
                        x=1240.6,
                        y=980.04,
                        width=234.4,
                        height=895.4
                    )
                )
            ]
            annotation = FrameAnnotations(
                source=SourceMetadata("test-source-id", (1920, 1080)),
                index=self.frame_index,
                annotations=annotations_list
            )

            self.frame_index += 1

        return annotation

@pytest.fixture
def consumer():
    """Fixture to provide a mock callback."""
    return MagicMock()


@pytest.mark.unit
@pytest.mark.parametrize("n_annotations", [0, 1, 5, 10])
def test_annotation_streamer_produces_and_feeds_expected_number_of_annotations(n_annotations, consumer):
    """Tests that the AnnotationStreamer produces and feeds the correct number of annotations."""
    # arrange
    streamer = DummyAnnotationStreamer(n_annotations, consumer)

    # act
    streamer.start_streaming()
    streamer.wait_for_completion()

    # assert
    assert consumer.feed.call_count == n_annotations + 1

    for i in range(n_annotations):
        call_args = consumer.feed.call_args_list[i]
        annotation = call_args[0][0]
        assert isinstance(annotation, FrameAnnotations)

    assert consumer.feed.call_args_list[n_annotations][0][0] is None

    assert streamer.get_status() == StreamerStatus.COMPLETED


@pytest.mark.unit
def test_annotation_streamer_should_indicate_stopping_when_stopped_early(consumer):
    """Tests that the AnnotationStreamer indicates that it was stopped when stopped early."""
    # arrange
    streamer = DummyAnnotationStreamer(100, consumer)
    streamer.start_streaming()

    # act
    streamer.stop_streaming()

    # assert
    assert streamer.get_status() == StreamerStatus.STOPPED


@pytest.mark.unit
def test_annotation_streamer_should_normalize_bboxes(consumer):
    """Tests that the AnnotationStreamer normalizes bboxes when provided a normalizer."""
    # arrange
    n_annotations = 3
    normalizer = SimpleBBoxNormalizer((0, 1))
    streamer = DummyAnnotationStreamer(n_annotations, consumer, normalizer)

    # act
    streamer.start_streaming()
    streamer.wait_for_completion()

    # assert
    assert consumer.feed.call_count == 4
    for i in range(n_annotations):
        call_args = consumer.feed.call_args_list[i]
        annotation = call_args[0][0]
        for anno_bbox in annotation.annotations:
            assert anno_bbox.bbox.x < 1
            assert anno_bbox.bbox.x > 0
            assert anno_bbox.bbox.y < 1
            assert anno_bbox.bbox.y > 0