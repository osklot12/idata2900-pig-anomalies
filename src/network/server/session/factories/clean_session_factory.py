from src.network.server.session.factories.session_factory import SessionFactory, T
from src.network.server.session.session import Session


class SimpleSessionFactory(SessionFactory):
    """Simple session factory."""

    def create_session(self, client_address: str) -> Session[T]:
        pass