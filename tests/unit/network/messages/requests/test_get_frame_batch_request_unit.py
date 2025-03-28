from unittest.mock import Mock

import pytest

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataset_split import DatasetSplit
from src.network.messages.requests.get_frame_batch_request import GetFrameBatchRequest
from src.network.messages.responses.frame_batch_response import FrameBatchResponse
from tests.utils.generators.dummy_annotations_generator import DummyAnnotationsGenerator
from tests.utils.generators.dummy_frame_generator import DummyFrameGenerator


@pytest.fixture
def batch():
    """Fixture to provide a batch of annotated frames."""
    return [
        AnnotatedFrame(
            frame=DummyFrameGenerator.generate(80, 60),
            annotations=DummyAnnotationsGenerator.generate(3)
        )
    ]


@pytest.fixture
def instance_provider(batch):
    """Fixture to provide a FrameInstanceProvider instance."""
    provider = Mock()
    provider.get_batch.return_value = batch

    return provider


@pytest.mark.unit
def test_initialize_with_invalid_batch_size():
    """Tests that passing a negative batch_size upon construction will raise."""
    # act & assert
    with pytest.raises(ValueError, match="batch_size must be greater than 0"):
        GetFrameBatchRequest(DatasetSplit.TRAIN, 0)


@pytest.mark.unit
def test_execution_returns_frame_batch_response(instance_provider, batch):
    """Tests that execution of the request successfully returns a valid FrameBatchResponse."""
    # arrange
    request = GetFrameBatchRequest(DatasetSplit.TRAIN, 1)

    # act
    result = request.execute(instance_provider)

    # assert
    assert isinstance(result, FrameBatchResponse)
