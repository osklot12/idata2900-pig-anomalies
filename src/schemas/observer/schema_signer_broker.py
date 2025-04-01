from typing import TypeVar, Generic

from src.schemas.observer.broker import Broker
from src.schemas.observer.schema_broker import SchemaBroker
from src.schemas.schema import Schema
from src.schemas.signed_schema import SignedSchema

T = TypeVar("T", bound=Schema)

class SchemaSignerBroker(Generic[T], Broker[T]):
    """SchemaBroker that signs schemas."""

    def __init__(self, issuer_id: str):
        """
        Initializes the SignedSchemaBroker.

        Args:
            issuer_id: the id of the issuer
        """
        self._id = issuer_id
        self._broker = SchemaBroker[SignedSchema[T]]()

    def notify(self, schema: T) -> None:
        self._broker.notify(SignedSchema(issuer_id=self._id, schema=schema))

    def get_schema_broker(self) -> SchemaBroker[SignedSchema[T]]:
        """Returns the underlying SchemaBroker instance."""
        return self._broker