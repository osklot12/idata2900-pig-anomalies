from typing import List

from src.schemas.algorithms.demand_estimator import DemandEstimator
from src.schemas.schemas.pressure_schema import PressureSchema


class SimpleDemandEstimator(DemandEstimator):
    """Simple demand estimator."""

    def estimate(self, schemas: List[PressureSchema]) -> float:
        if not schemas:
            raise ValueError("schemas cannot be empty or None")

        if schemas[-1].usage < 1:
            result = 2
        else:
            result = pow(2, -self._get_avg_balance(schemas))

        return result

    @staticmethod
    def _get_avg_balance(schemas: List[PressureSchema]) -> float:
        """Computes the average in/out balance for the pressure schemas."""
        if not schemas:
            raise ValueError("Cannot compute balance, no pressure schemas available")

        numerator = 0
        denominator = 0
        for point in schemas:
            numerator += point.inputs - point.outputs
            denominator += point.inputs + point.outputs

        if denominator == 0:
            result = 0
        else:
            result = numerator / denominator

        return result