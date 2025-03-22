from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np

class DDPSenderInterface(ABC):
    """Abstract interface for sending frame + annotation data over DDP"""

    @abstractmethod
    def connect_to_workers(self):
        """Establish connections with worker PCs."""
        pass

    @abstractmethod
    def send_data(self, frames: List[np.ndarray], annotations: List[List[Dict]]):
        """Send batch of frames with corresponding annotations."""
        pass

    @abstractmethod
    def disconnect_workers(self):
        """Close connections properly."""
        pass
