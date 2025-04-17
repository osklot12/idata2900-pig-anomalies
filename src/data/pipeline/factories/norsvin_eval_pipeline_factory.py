from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataclasses.compressed_annotated_frame import CompressedAnnotatedFrame
from src.data.pipeline.factories.pipeline_factory import PipelineFactory
from src.data.pipeline.field_transformer import FieldTransformer
from src.data.pipeline.norsvin_pipeline_config import RESIZE_SHAPE, NORMALIZE_RANGE
from src.data.pipeline.pipeline import Pipeline
from src.data.pipeline.pipeline_tail import PipelineTail
from src.data.pipeline.preprocessor import Preprocessor
from src.data.processing.bbox_normalizer_processor import BBoxNormalizerProcessor
from src.data.processing.frame_resizer import FrameResizer
from src.data.processing.normalization.simple_bbox_normalizer import SimpleBBoxNormalizer
from src.data.processing.zlib_compressor import ZlibCompressor


class NorsvinEvalPipelineFactory(PipelineFactory[AnnotatedFrame, CompressedAnnotatedFrame]):
    """Factory for creating evaluation set pipelines for the Norsvin dataset."""

    def create_pipeline(self) -> PipelineTail[AnnotatedFrame, CompressedAnnotatedFrame]:
        return Pipeline(
            FieldTransformer.of("frame").using(FrameResizer(RESIZE_SHAPE))
        ).then(
            Preprocessor(BBoxNormalizerProcessor(SimpleBBoxNormalizer(NORMALIZE_RANGE)))
        ).then(
            Preprocessor(ZlibCompressor())
        )