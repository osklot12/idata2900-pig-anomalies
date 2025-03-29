import pytest

from src.network.messages.requests.handlers.request_handler import RequestHandler
from src.network.messages.requests.handlers.registry.simple_request_handler_registry import SimpleRequestHandlerRegistry
from src.network.messages.requests.request import Request
from src.network.messages.responses.response import Response


class DummyRequest(Request):
    """Dummy request for testing."""
    pass

class DummyResponse(Response):
    """Dummy response for testing."""
    pass

class DummyHandler(RequestHandler):
    """Dummy handler for testing."""

    def handle(self, request: Request) -> Response:
        return DummyResponse()


@pytest.mark.unit
def test_register():
    """Tests that register() successfully registers and maps a request type to a RequestHandler."""
    # arrange
    registry = SimpleRequestHandlerRegistry()
    handler = DummyHandler()
    request = DummyRequest()

    # act
    registry.register(DummyRequest, handler)
    retrieved_handler = registry.get_handler(request)

    # assert
    assert retrieved_handler is handler