from src.data.structures.runtime_param import RuntimeParam
import random


class RandomFloat(RuntimeParam[float]):
    """Computes a random uniform float at request."""

    def __init__(self, min_value: float, max_value: float):
        """
        Initializes a RandomFloat instance.

        Args:
            min_value (float): minimum value
            max_value (float): maximum value
        """
        if min_value > max_value:
            raise ValueError("min_value cannot be greater than max_value")

        self._min_value = min_value
        self._max_value = max_value

    def resolve(self) -> float:
        return random.uniform(self._min_value, self._max_value)

    def __repr__(self):
        return f"RandomFloat(min={self._min_value}, max={self._max_value})"