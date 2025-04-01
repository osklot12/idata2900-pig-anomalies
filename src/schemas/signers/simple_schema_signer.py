import time

from src.schemas.signed_schema import SignedSchema
from src.schemas.signers.schema_signer import SchemaSigner, T


class SimpleSchemaSigner(SchemaSigner):
    """A simple schema signer."""

    def __init__(self, signature: str):
        """
        Initializes a SimpleSchemaSigner instance.

        Args:
            signature (str): the signature to sign with
        """
        self._signature = signature

    def sign(self, schema: T) -> SignedSchema[T]:
        return SignedSchema(signature=self._signature, timestamp=time.time(), schema=schema)