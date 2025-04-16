import pytest

from src.data.dataclasses.source_metadata import SourceMetadata
from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.preprocessing.augmentation.augmentors.annotated_frame_augmentor import AnnotatedFrameAugmentor
from tests.utils.generators.dummy_annotations_generator import DummyAnnotationsGenerator
from tests.utils.generators.dummy_frame_generator import DummyFrameGenerator


@pytest.fixture
def frame():
    """Fixture to provide a dummy frame."""
    return DummyFrameGenerator.generate(640, 640)


@pytest.fixture
def annotations():
    """Fixture to provide dummy annotations."""
    return DummyAnnotationsGenerator.generate(10)


@pytest.mark.unit
def test_augment(frame, annotations):
    """Tests that the instance is augmented without raising."""
    # arrange
    instance = AnnotatedFrame(
        source=SourceMetadata(source_id="test-source-id", frame_resolution=(640, 640)),
        index=0,
        frame=frame,
        annotations=annotations
    )
    aug = AnnotatedFrameAugmentor()

    # act
    augmented = aug.process(instance)

    # assert
    assert instance is not augmented
