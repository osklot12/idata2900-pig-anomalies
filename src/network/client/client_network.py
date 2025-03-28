import socket
from typing import TypeVar
from src.network.messages.requests.request import Request
from src.network.messages.response.response import Response
from src.network.message_handler import MessageHandler

T = TypeVar("T", bound=Response)


class NetworkClient:
    def __init__(self, ip: str, port: int):
        self._ip = ip
        self._port = port
        self._socket: socket.socket | None = None
        self._msg_len = NETWORK_MSG_LEN_BYTES

    def connect(self) -> None:
        """
        Establishes a persistent connection to the server.
        """
        if self._socket:
            raise RuntimeError("Already connected")

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self._ip, self._port))

    def send_request(self, request: Request) -> T:
        """
        Sends a request over the open socket and returns the response.

        Args:
            request (Request): The request to send.

        Returns:
            T: The deserialized response from the server.
        """
        if self._socket is None:
            raise RuntimeError("Not connected")

        MessageHandler.send_message(self._socket, request)

        response: T = MessageHandler.receive_message(self._socket)
        return response

    def close(self) -> None:
        """
        Closes the persistent connection.
        """
        if self._socket:
            self._socket.close()
            self._socket = None
