from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np

from src.data.dataclasses.annotated_frame import AnnotatedFrame


class DDPSenderInterface(ABC):
    """Abstract interface for sending frame + annotation data over DDP"""

    @abstractmethod
    def connect_to_workers(self) -> None:
        """Establish connections with worker PCs."""
        raise NotImplementedError

    @abstractmethod
    def send_data(self, data_batch: List[AnnotatedFrame]) -> None:
        """
        Sends a batch of annotated frames.

        Args:
            data_batch (List[AnnotatedFrame]): list of annotated frames
        """
        raise NotImplementedError

    @abstractmethod
    def disconnect_workers(self) -> None:
        """Close connections properly."""
        raise NotImplementedError
