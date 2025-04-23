from typing import List

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataclasses.compressed_annotated_frame import CompressedAnnotatedFrame
from src.data.pipeline.field_transformer import FieldTransformer
from src.data.pipeline.filter_component import FilterComponent
from src.data.pipeline.norsvin_pipeline_config import RESIZE_SHAPE, NORMALIZE_RANGE, CLASS_COUNTS
from src.data.pipeline.pipeline import Pipeline
from src.data.pipeline.factories.pipeline_factory import PipelineFactory
from src.data.pipeline.pipeline_tail import PipelineTail
from src.data.pipeline.preprocessor import Preprocessor
from src.data.pipeline.splitting_preprocessor import SplittingPreprocessor
from src.data.processing.augmentation.augmentation_plan_factory import AugmentationPlanFactory
from src.data.processing.augmentation.photometric.brightness_filter import BrightnessFilter
from src.data.processing.augmentation.photometric.color_jitter_filter import ColorJitterFilter
from src.data.processing.augmentation.photometric.contrast_filter import ContrastFilter
from src.data.processing.augmentation.photometric.gaussian_noise_filter import GaussianNoiseFilter
from src.data.processing.augmentation.photometric.photometric_filter import PhotometricFilter
from src.data.processing.augmentor import Augmentor
from src.data.processing.bbox_normalizer_processor import BBoxNormalizerProcessor
from src.data.processing.class_balancer import ClassBalancer
from src.data.processing.cond_multiplier import CondMultiplier
from src.data.processing.frame_resizer import FrameResizer
from src.data.processing.normalization.simple_bbox_normalizer import SimpleBBoxNormalizer
from src.data.processing.zlib_compressor import ZlibCompressor
from src.data.structures.random_float import RandomFloat

class NorsvinTrainPipelineFactory(PipelineFactory[AnnotatedFrame, CompressedAnnotatedFrame]):
    """Factory for creating training set pipelines for the Norsvin dataset."""

    def create_pipeline(self) -> PipelineTail[AnnotatedFrame, CompressedAnnotatedFrame]:
        def is_annotated(frame: AnnotatedFrame) -> bool:
            return len(frame.annotations) > 0

        return Pipeline(
            FilterComponent(is_annotated)
        ).then(
            FieldTransformer.of("frame").using(FrameResizer(RESIZE_SHAPE))
        ).then(
            Preprocessor(BBoxNormalizerProcessor(SimpleBBoxNormalizer(NORMALIZE_RANGE)))
        ).then(
            SplittingPreprocessor(ClassBalancer(class_counts=CLASS_COUNTS, max_samples_per=3))
        ).then(
            SplittingPreprocessor(CondMultiplier(n=2, condition=is_annotated))
        ).then(
            Preprocessor(Augmentor(plan_factory=AugmentationPlanFactory(), filters=self._create_filters()))
        ).then(
            Preprocessor(ZlibCompressor())
        )

    @staticmethod
    def _create_filters() -> List[PhotometricFilter]:
        """Creates photometric filters for augmentation."""
        return [
            BrightnessFilter(beta=RandomFloat(-30, 30)),
            ContrastFilter(alpha=RandomFloat(0.8, 1.2)),
            ColorJitterFilter(saturation_scale=RandomFloat(0.8, 1.2), hue_shift=RandomFloat(-10, 10)),
            GaussianNoiseFilter(std=RandomFloat(5, 15))
        ]