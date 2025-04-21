import time
from typing import TypeVar, Generic

from src.data.dataset.dataset_split import DatasetSplit
from src.network.server.session.factories.session_factory import SessionFactory
from src.network.server.session.session import Session

T = TypeVar("T")


class CleanSessionFactory(Generic[T], SessionFactory[T]):
    """Factory for creating clean Sessions instances without any existing streams."""

    def create_session(self, client_address: str) -> Session[T]:
        return Session(
            client_address=client_address,
            created_at=time.time(),
            streams={split: None for split in DatasetSplit}
        )