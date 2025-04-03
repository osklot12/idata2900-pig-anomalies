from typing import TypeVar, Generic

from src.schemas.schemas.schema import Schema
from src.schemas.schemas.schema_listener import SchemaListener
from src.schemas.schemas.signed_schema import SignedSchema
from src.schemas.signing.schema_signer import SchemaSigner

T = TypeVar("T", bound=Schema)

class Sign(Generic[T], SchemaListener[T]):
    """Signs incoming schemas as passed them forward."""

    def __init__(self, consumer: SchemaListener[SignedSchema[T]], signer: SchemaSigner):
        """
        Initializes a Sign instance.

        Args:
            consumer (SchemaListener[SignedSchema[T]]): the consumer to forward signed schemas to
            signer (SchemaSigner): the schema signer to use
        """
        self._consumer = consumer
        self._signer = signer

    def new_schema(self, schema: T) -> None:
        self._consumer.new_schema(self._signer.sign(schema))