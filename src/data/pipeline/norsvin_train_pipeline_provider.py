from typing import Optional

from src.data.compressors.zlib_compressor import ZlibCompressor
from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataclasses.compressed_annotated_frame import CompressedAnnotatedFrame
from src.data.pipeline.consumer import Consumer
from src.data.pipeline.consumer_provider import ConsumerProvider
from src.data.pipeline.field_transformer import FieldTransformer
from src.data.pipeline.pipeline import Pipeline
from src.data.pipeline.preprocessor import Preprocessor
from src.data.preprocessing.augmentation.augmentors.augmentor_component import AugmentorComponent
from src.data.preprocessing.augmentation.augmentors.instance_augmentor import InstanceAugmentor
from src.data.preprocessing.augmentation.augmentors.photometric.factories.brightness_filter_factory import \
    BrightnessFilterFactory
from src.data.preprocessing.augmentation.augmentors.photometric.factories.color_jitter_filter_factory import \
    ColorJitterFilterFactory
from src.data.preprocessing.augmentation.augmentors.photometric.factories.constrast_filter_factory import \
    ContrastFilterFactory
from src.data.preprocessing.augmentation.augmentors.photometric.factories.gaussian_noise_filter_factory import \
    GaussianNoiseFilterFactory
from src.data.preprocessing.augmentation.cond_multiplier_component import CondMultiplierComponent
from src.data.preprocessing.augmentation.plan.augmentation_plan_factory import AugmentationPlanFactory
from src.data.processing.class_balancer import ClassBalancer
from src.data.preprocessing.normalization.normalizers.bbox_normalizer_component import BBoxNormalizerComponent
from src.data.preprocessing.normalization.normalizers.simple_bbox_normalizer import SimpleBBoxNormalizer
from src.data.processing.bbox_normalizer_processor import BBoxNormalizerProcessor
from src.data.processing.frame_resizer import FrameResizer
from src.data.structures.atomic_bool import AtomicBool
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass

RESIZE_SHAPE = (640, 640)
NORMALIZE_RANGE = (0, 1)


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

        sink = self._sink_provider.get_consumer(release=release)
        if sink is not None:
            result = Pipeline(
                FieldTransformer.of("frame").using(FrameResizer(RESIZE_SHAPE))
            ).then(
                Preprocessor(BBoxNormalizerProcessor(SimpleBBoxNormalizer(NORMALIZE_RANGE)))
            ).then(
                self._create_balancer()
            ).then(
                self._create_multiplier()
            ).then(
                self._create_augmentor()
            ).then(
                self._create_compressor()
            ).into(sink)

        return result

    @staticmethod
    def _create_balancer() -> ClassBalancer:
        """Creates and returns a ClassBalancer instance."""
        return ClassBalancer(
            class_counts={
                NorsvinBehaviorClass.BELLY_NOSING: 1885,
                NorsvinBehaviorClass.TAIL_BITING: 1073,
                NorsvinBehaviorClass.EAR_BITING: 1008,
                NorsvinBehaviorClass.TAIL_DOWN: 1107
            },
            max_samples_per=3
        )

    @staticmethod
    def _create_multiplier() -> CondMultiplierComponent[AnnotatedFrame]:
        """Creates and returns a CondMultiplierComponent instance."""

        def is_annotated(instance: AnnotatedFrame) -> bool:
            """Predicate for checking whether the instance is annotated."""
            return len(instance.annotations) > 0

        return CondMultiplierComponent(
            n=2,
            predicate=is_annotated
        )

    @staticmethod
    def _create_augmentor() -> AugmentorComponent:
        """Creates and returns an AugmentorComponent instance."""
        return AugmentorComponent(
            InstanceAugmentor(
                plan_factory=AugmentationPlanFactory(),
                filters=[
                    BrightnessFilterFactory(beta_range=(-30, 30)).create_filter(),
                    ContrastFilterFactory(alpha_range=(0.8, 1.2)).create_filter(),
                    ColorJitterFilterFactory(saturation_range=(0.8, 1.2), hue_range=(-10, 10)).create_filter(),
                    GaussianNoiseFilterFactory(std_range=(5, 15)).create_filter()
                ]
            )
        )

    @staticmethod
    def _create_compressor() -> ZlibCompressor:
        """Creates and returns a ZlibCompressor instance."""
        return ZlibCompressor()
