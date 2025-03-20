from typing import List

from src.data.dataclasses.bbox_annotation import BBoxAnnotation
from src.data.dataclasses.bounding_box import BoundingBox
from src.data.dataclasses.frame_annotation import FrameAnnotation
from src.data.loading.loaders.video_annotations_loader import VideoAnnotationsLoader
from tests.utils.test_annotation_class import TestAnnotationLabel


class DummyVideoAnnotationsLoader(VideoAnnotationsLoader):
    """A dummy video annotations loader for testing."""

    def load_video_annotations(self, annotations_id: str) -> List[FrameAnnotation]:
        return [
            FrameAnnotation(
                source="test-source",
                index=0,
                annotations=[
                    BBoxAnnotation(
                        TestAnnotationLabel.CODING,
                        BoundingBox(320, 540, 103, 80)
                    )
                ],
                end_of_stream=False
            ),
            FrameAnnotation(
                source="test-source",
                index=1,
                annotations=[

                ],
                end_of_stream=False
            ),
            FrameAnnotation(
                source="test-source",
                index=2,
                annotations=[
                    BBoxAnnotation(
                        TestAnnotationLabel.CODING,
                        BoundingBox(1450, 980, 200, 79)
                    ),
                    BBoxAnnotation(
                        TestAnnotationLabel.DEBUGGING,
                        BoundingBox(1123, 1201, 163, 178)
                    )
                ],
                end_of_stream=False
            )
        ]