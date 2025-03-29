from unittest.mock import Mock

import pytest

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataset.dataset_split import DatasetSplit
from src.network.messages.requests.get_frame_batch_request import GetFrameBatchRequest
from src.network.messages.requests.handlers.get_frame_batch_handler import GetFrameBatchHandler
from src.network.messages.responses.frame_batch_response import FrameBatchResponse
from tests.utils.generators.dummy_annotations_generator import DummyAnnotationsGenerator
from tests.utils.generators.dummy_frame_generator import DummyFrameGenerator


@pytest.fixture
def batch():
    """Fixture to provide a batch of annotated frames."""
    return [
        AnnotatedFrame(
            frame=DummyFrameGenerator.generate(80, 60),
            annotations=DummyAnnotationsGenerator.generate(10)
        )
    ]


@pytest.fixture
def instance_provider(batch):
    """Fixture to provide a FrameInstanceProvider instance."""
    provider = Mock()
    provider.get_batch.return_value = batch
    return provider


@pytest.mark.unit
def test_handle_returns_expected_response(instance_provider, batch):
    """Tests that handle() returns the expected response."""
    # arrange
    handler = GetFrameBatchHandler(instance_provider)
    request = GetFrameBatchRequest(split=DatasetSplit.TRAIN, batch_size=10)

    # act
    response = handler.handle(request)

    # assert
    assert isinstance(response, FrameBatchResponse)
    assert response.batch == batch
    instance_provider.get_batch.assert_called_once_with(DatasetSplit.TRAIN, 10)
