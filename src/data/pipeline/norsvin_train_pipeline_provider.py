from typing import Optional, List

from src.data.compressors.zlib_compressor import ZlibCompressor
from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataclasses.compressed_annotated_frame import CompressedAnnotatedFrame
from src.data.pipeline.consumer import Consumer
from src.data.pipeline.consumer_provider import ConsumerProvider
from src.data.pipeline.field_transformer import FieldTransformer
from src.data.pipeline.pipeline import Pipeline
from src.data.pipeline.preprocessor import Preprocessor
from src.data.pipeline.splitting_preprocessor import SplittingPreprocessor
from src.data.preprocessing.augmentation.augmentors.photometric.brightness_filter import BrightnessFilter
from src.data.preprocessing.augmentation.augmentors.photometric.color_jitter_filter import ColorJitterFilter
from src.data.preprocessing.augmentation.augmentors.photometric.contrast_filter import ContrastFilter
from src.data.preprocessing.augmentation.augmentors.photometric.gaussian_noise_filter import GaussianNoiseFilter
from src.data.preprocessing.augmentation.augmentors.photometric.photometric_filter import PhotometricFilter
from src.data.preprocessing.augmentation.plan.augmentation_plan_factory import AugmentationPlanFactory
from src.data.processing.augmentor import Augmentor
from src.data.processing.class_balancer import ClassBalancer
from src.data.preprocessing.normalization.normalizers.simple_bbox_normalizer import SimpleBBoxNormalizer
from src.data.processing.bbox_normalizer_processor import BBoxNormalizerProcessor
from src.data.processing.cond_multiplier import CondMultiplier
from src.data.processing.frame_resizer import FrameResizer
from src.data.structures.atomic_bool import AtomicBool
from src.data.structures.random_float import RandomFloat
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass

RESIZE_SHAPE = (640, 640)
NORMALIZE_RANGE = (0, 1)
CLASS_COUNTS = {
    NorsvinBehaviorClass.BELLY_NOSING: 1885,
    NorsvinBehaviorClass.TAIL_BITING: 1073,
    NorsvinBehaviorClass.EAR_BITING: 1008,
    NorsvinBehaviorClass.TAIL_DOWN: 1107
}


class NorsvinTrainPipelineProvider(ConsumerProvider[AnnotatedFrame]):
    """Provider of Norsvin training set pipelines."""

    def __init__(self, sink_provider: ConsumerProvider[CompressedAnnotatedFrame]):
        """
        Initializes a NorsvinTrainPipelineProvider instance.

        Args:
            sink_provider (ConsumerProvider[CompressedAnnotatedFrame]): provider of endpoint (sink) for the pipeline
        """
        self._sink_provider = sink_provider

    def get_consumer(self, release: Optional[AtomicBool] = None) -> Optional[Consumer[AnnotatedFrame]]:
        result = None

        def is_annotated(frame: AnnotatedFrame) -> bool:
            return len(frame.annotations) > 0

        sink = self._sink_provider.get_consumer(release=release)
        if sink is not None:
            result = Pipeline(
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
                self._create_compressor()
            ).into(sink)

        return result

    @staticmethod
    def _create_filters() -> List[PhotometricFilter]:
        """Creates photometric filters for augmentation."""
        return [
            BrightnessFilter(beta=RandomFloat(-30, 30)),
            ContrastFilter(alpha=RandomFloat(0.8, 1.2)),
            ColorJitterFilter(saturation_scale=RandomFloat(0.8, 1.2), hue_shift=RandomFloat(-10, 10)),
            GaussianNoiseFilter(std=RandomFloat(5, 15))
        ]

    @staticmethod
    def _create_compressor() -> ZlibCompressor:
        """Creates and returns a ZlibCompressor instance."""
        return ZlibCompressor()
