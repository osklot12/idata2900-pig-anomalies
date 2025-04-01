from dataclasses import dataclass

from src.schemas.schemas.schema import Schema

@dataclass(frozen=True)
class TimestampedSchema(Schema):
    """
    A timestamped schema.

    Attributes:
        timestamp (float): the unix timestamp for the schema
    """
    timestamp: float