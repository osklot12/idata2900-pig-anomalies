from unittest.mock import Mock

import pytest

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataset.dataset_split import DatasetSplit
from src.data.errors.data_retrieval_error import DataRetrievalError
from src.network.client.network_client import NetworkClient
from src.network.messages.responses.frame_batch_response import FrameBatchResponse
from src.network.network_frame_instance_provider import NetworkFrameInstanceProvider
from tests.utils.generators.dummy_annotations_generator import DummyAnnotationsGenerator
from tests.utils.generators.dummy_frame_generator import DummyFrameGenerator


@pytest.fixture
def batch():
    """A fixture to provide a batch of annotated frames."""
    return [
        AnnotatedFrame(
            frame=DummyFrameGenerator.generate(80, 60),
            annotations=DummyAnnotationsGenerator.generate(10)
        )
    ]


@pytest.fixture
def client(batch):
    """Fixture to provide a NetworkClient instance."""
    client = Mock(spec=NetworkClient)
    client.send_request.return_value = FrameBatchResponse(batch)
    return client


@pytest.mark.unit
def test_get_batch_returns_correct_batch(client, batch):
    """Tests that get_batch() returns the correct batch."""
    # arrange
    provider = NetworkFrameInstanceProvider(client)

    # act
    result = provider.get_batch(split=DatasetSplit.TRAIN, batch_size=10)

    # assert
    assert result == batch
    client.send_request.assert_called_once()


@pytest.mark.unit
def test_get_batch_raises_data_retrieval_error_on_send_exception():
    """Tests that get_batch() raises DataRetrievalError when the client raises an exception."""
    # arrange
    client = Mock(spec=NetworkClient)
    client.send_request.side_effect = RuntimeError("Connection failed")

    provider = NetworkFrameInstanceProvider(client)

    # act & assert
    with pytest.raises(DataRetrievalError, match="Failed to send request: Connection failed"):
        provider.get_batch(split=DatasetSplit.TRAIN, batch_size=10)


def test_get_batch_raises_runtime_error_on_unexpected_response():
    """Tests that get_batch() raises DataRetrievalError if the response is of unexpected type."""
    # arrange
    client = Mock(spec=NetworkClient)
    client.send_request.return_value = object()

    provider = NetworkFrameInstanceProvider(client)

    # act & assert
    with pytest.raises(DataRetrievalError, match="Unexpected response type"):
        provider.get_batch(split=DatasetSplit.TRAIN, batch_size=10)