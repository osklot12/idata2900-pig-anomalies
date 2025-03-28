from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.network.client.client_context import ClientContext
from src.network.messages.response.response import Response, T


class FrameInstanceResponse(Response):
    def __init__(self, instance: AnnotatedFrame):
        """
        Initialize a FrameInstanceResponse object.

        Args:
            instance (AnnotatedFrame): Frame instance.
        """
        self._instance = instance

    def execute(self, context: ClientContext) -> T:
        return self._instance
