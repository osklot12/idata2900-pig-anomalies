from unittest.mock import Mock

import pytest

from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.network_stream import NetworkStream
from src.network.messages.responses.get_batch_response import GetBatchResponse
from src.network.messages.responses.open_stream_response import OpenStreamResponse
from src.network.messages.responses.response_status import ResponseStatus


@pytest.fixture
def batch():
    """Fixture to provide a batch of dummy data."""
    return [f"string_{i}" for i in range(8)]


@pytest.fixture
def client(batch):
    """Fixture to provide a mock NetworkClient instance."""
    client = Mock()

    # First call returns OpenStreamResponse, second call returns GetBatchResponse
    client.send_request.side_effect = [
        OpenStreamResponse(status=ResponseStatus.SUCCESS),
        GetBatchResponse(status=ResponseStatus.SUCCESS, batch=batch)
    ]
    return client


@pytest.mark.unit
def test_read_returns_batch_successfully(client, batch):
    """Tests that calling read returns the expected batch."""
    # arrange
    stream = NetworkStream[str](client=client, split=DatasetSplit.TRAIN, batch_type=str, batch_size=8)

    # arrange
    received = stream.read()

    # assert
    assert received == batch


@pytest.mark.unit
def test_read_returns_unexpected_batch_type_raises(client, batch):
    """Tests that reading an unexpected batch type raises an exception."""
    # arrange
    stream = NetworkStream[int](client=client, split=DatasetSplit.TRAIN, batch_type=int, batch_size=8)

    # act & assert
    with pytest.raises(RuntimeError):
        stream.read()