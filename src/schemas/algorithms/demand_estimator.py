from abc import ABC, abstractmethod
from typing import List

from src.schemas.technical.pressure_schema import PressureSchema


class DemandEstimator(ABC):
    """Interface for demand estimators."""

    @abstractmethod
    def estimate(self, schemas: List[PressureSchema]) -> float:
        """
        Estimates the demand based on pressure.

        Args:
            schemas (List[PressureSchema]): list of pressure schemas

        Returns:
            float: a value between 0 and infinity, being a scalar for the current production to reach desired demand
        """
        raise NotImplementedError