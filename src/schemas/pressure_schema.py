from dataclasses import dataclass

from src.schemas.schema import Schema


@dataclass(frozen=True)
class PressureSchema(Schema):
    """
    A schema for component pressure information.

    Attributes:
        inputs (int): the number of inputs
        outputs (int): the number of outputs
        occupied (float): the occupied percentage (0 - 1 range)
        timestamp (float): the time of the measured pressure (unix)
    """
    inputs: int
    outputs: int
    occupied: float
    timestamp: float