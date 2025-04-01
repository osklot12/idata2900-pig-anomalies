from unittest.mock import MagicMock

import pytest
import time

from src.schemas.schemas.schema import Schema
from src.schemas.signing.simple_schema_signer import SimpleSchemaSigner


@pytest.fixture
def signature():
    """Fixture to provide the string signature."""
    return "test-signature"


@pytest.fixture
def signer(signature):
    """Fixture to provide a SimpleSchemaSigner instance."""
    return SimpleSchemaSigner(signature=signature)


@pytest.fixture
def schema():
    """Fixture to provide a mock Schema instance."""
    return MagicMock(spec=Schema)


@pytest.mark.unit
def test_schema_gets_signed_successfully(signature, signer, schema):
    """Tests that the signer successfully signs a schema."""
    # act
    signed_schema = signer.sign(schema)

    # assert
    assert signed_schema.signature == signature
    assert signed_schema.schema == schema