from dataclasses import dataclass

from src.schemas.schemas.timestamped_schema import TimestampedSchema


@dataclass(frozen=True)
class PressureSchema(TimestampedSchema):
    """
    A schema for component pressure information.

    Attributes:
        inputs (int): the number of inputs
        outputs (int): the number of outputs
        usage (float): the usage percentage (0 - 1 range)
    """
    inputs: int
    outputs: int
    usage: float