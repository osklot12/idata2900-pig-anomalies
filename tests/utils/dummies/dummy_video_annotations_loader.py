from typing import List

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.bbox import BBox
from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.loading.loaders.video_annotations_loader import VideoAnnotationsLoader
from tests.utils.test_annotation_label import TestAnnotationLabel


class DummyVideoAnnotationsLoader(VideoAnnotationsLoader):
    """A dummy video annotations loader for testing."""

    def load_video_annotations(self, annotations_id: str) -> List[FrameAnnotations]:
        return [
            FrameAnnotations(
                source="test-source",
                index=0,
                annotations=[
                    AnnotatedBBox(
                        TestAnnotationLabel.CODING,
                        BBox(320, 540, 103, 80)
                    )
                ],
                end_of_stream=False
            ),
            FrameAnnotations(
                source="test-source",
                index=1,
                annotations=[

                ],
                end_of_stream=False
            ),
            FrameAnnotations(
                source="test-source",
                index=2,
                annotations=[
                    AnnotatedBBox(
                        TestAnnotationLabel.CODING,
                        BBox(1450, 980, 200, 79)
                    ),
                    AnnotatedBBox(
                        TestAnnotationLabel.DEBUGGING,
                        BBox(1123, 1201, 163, 178)
                    )
                ],
                end_of_stream=False
            )
        ]