from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataclasses.bbox import BBox
from src.data.processing.processor import Processor, I, O


class BBoxDenormalizerProcessor(Processor[AnnotatedFrame, AnnotatedFrame]):
    """Denormalizes bounding boxes."""

    def process(self, data: AnnotatedFrame) -> AnnotatedFrame:
        height, width = data.frame.shape[:2]
        denormalized_annotations = [
            AnnotatedBBox(
                cls=anno.cls,
                bbox=BBox(
                    x=anno.bbox.x * width,
                    y=anno.bbox.y * height,
                    width=anno.bbox.width * width,
                    height=anno.bbox.height * height,
                )
            )
            for anno in data.annotations
        ]

        return AnnotatedFrame(
            source=data.source,
            index=data.index,
            frame=data.frame,
            annotations=denormalized_annotations
        )