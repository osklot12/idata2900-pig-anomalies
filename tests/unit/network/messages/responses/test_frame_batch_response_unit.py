import pytest

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.network.messages.responses.frame_batch_response import FrameBatchResponse
from tests.utils.generators.dummy_annotations_generator import DummyAnnotationsGenerator
from tests.utils.generators.dummy_frame_generator import DummyFrameGenerator


@pytest.fixture
def batch():
    """Fixture to provide a list of annotated frames."""
    return [
        AnnotatedFrame(
            frame=DummyFrameGenerator.generate(80, 60),
            annotations=DummyAnnotationsGenerator.generate(10)
        )
    ]


@pytest.mark.unit
def test_execute_returns_expected_batch(batch):
    """Tests that execution of the response returns the expected batch."""
    # arrange
    response = FrameBatchResponse(batch)

    # act
    result = response.execute()

    # assert
    assert result == batch
