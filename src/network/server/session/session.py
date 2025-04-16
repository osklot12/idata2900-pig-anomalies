from typing import Dict, TypeVar, Generic, Optional

from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.managed.managed_stream import ManagedStream

T = TypeVar("T")


class Session(Generic[T]):
    """Interface for network sessions."""

    def __init__(self, client_address: str, created_at: float, streams: Dict[DatasetSplit, Optional[ManagedStream[T]]]):
        """
        Initializes a Session instance.

        Args:
            client_address (str): the address of the client
            created_at (float): the time the session was created in UNIX format
            streams (Dict[DatasetSplit, ManagedStream[T]]): dictionary of dataset streams
        """
        self._client_address = client_address
        self._created_at = created_at
        self._streams: Dict[DatasetSplit, Optional[ManagedStream[T]]] = streams

    def get_client_address(self) -> str:
        """
        Returns the client address.

        Returns:
            str: the client address
        """
        return self._client_address

    def get_created_at(self) -> float:
        """
        Returns the time the session was created.

        Returns:
            float: the time the session was created
        """
        return self._created_at

    def get_stream(self, split: DatasetSplit) -> Optional[ManagedStream[T]]:
        """
        Returns the stream for the given split.

        Args:
            split (DatasetSplit): the split to get the stream for

        Returns:
            Optional[ManagedStream[T]]: the stream for the given split, or None if no stream was found
        """
        return self._streams.get(split, None)

    def set_stream(self, stream: Optional[ManagedStream[T]], split: DatasetSplit) -> None:
        """
        Sets the stream for the given split.

        Args:
            stream (Optional[ManagedStream[T]]): the stream to set
            split (DatasetSplit): the split to set the stream for
        """
        current_stream = self.get_stream(split)
        if current_stream is not None:
            current_stream.stop()

        self._streams[split] = stream

    def cleanup(self) -> None:
        """Cleans up resources."""
        for stream in self._streams.values():
            if stream is not None:
                stream.stop()