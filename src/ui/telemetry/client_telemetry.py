from abc import ABC, abstractmethod

class ClientTelemetry(ABC):
    """Displays metrics for network clients."""

    @abstractmethod
    def add_client(self, address: str) -> None:
        """
        Adds a client for the given address.

        Args:
            address (str): the client address
        """
        raise NotImplementedError

    @abstractmethod
    def remove_client(self, address: str) -> None:
        """
        Removes the client for the given address.

        Args:
            address (str): the address of client to remove
        """
        raise NotImplementedError

    @abstractmethod
    def report_request(self, client_address) -> None:
        """
        Reports a request made by the given client.

        Args:
            client_address (str): the address of client to report
        """
        raise NotImplementedError