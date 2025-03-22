from abc import ABC, abstractmethod

class DDPReceiverInterface(ABC):
    """Abstract interface for receiving and processing data."""

    @abstractmethod
    def start(self):
        """Start listening for incoming data."""
        pass

    @abstractmethod
    def handle_client(self, client_socket):
        """Process data received from the sender."""
        pass
