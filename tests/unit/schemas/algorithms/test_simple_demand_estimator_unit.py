import pytest
import time

from src.schemas.algorithms.simple_demand_estimator import SimpleDemandEstimator
from src.schemas.pressure_schema import PressureSchema


@pytest.fixture
def estimator():
    """Fixture to provide a SimpleDemandEstimator instance."""
    return SimpleDemandEstimator()


@pytest.mark.unit
def test_estimate_with_empty_schemas_raises(estimator):
    """Tests that estimating demand with an empty list of schemas will raise."""
    # act & assert
    with pytest.raises(ValueError, match="schemas cannot be empty or None"):
        estimator.estimate([])


@pytest.mark.unit
def test_estimate_high_demand_when_empty(estimator):
    """Tests that a high demand is estimated when the schemas report the full capacity is free."""
    # arrange
    schemas = [
        PressureSchema(inputs=10, outputs=5, usage=0.1, timestamp=time.time()),
        PressureSchema(inputs=10, outputs=5, usage=0, timestamp=time.time())
    ]

    # act
    result = estimator.estimate(schemas)

    # assert
    assert result == 2


@pytest.mark.unit
def test_estimate_high_demand_when_low_inputs_high_outputs(estimator):
    """Tests that a high demand is estimated when the schemas report high outputs and low inputs."""
    # arrange
    schemas = [
        PressureSchema(inputs=0, outputs=10, usage=0.1, timestamp=time.time()),
        PressureSchema(inputs=0, outputs=10, usage=0.2, timestamp=time.time()),
        PressureSchema(inputs=0, outputs=10, usage=1, timestamp=time.time())
    ]

    # act
    result = estimator.estimate(schemas)

    # assert
    assert result == 2


@pytest.mark.unit
def test_estimate_low_demand_when_high_inputs_low_outputs(estimator):
    """Tests that a low demand is estimated when the schemas report high inputs and low outputs."""
    # arrange
    schemas = [
        PressureSchema(inputs=10, outputs=0, usage=0.1, timestamp=time.time()),
        PressureSchema(inputs=10, outputs=0, usage=0.2, timestamp=time.time()),
        PressureSchema(inputs=10, outputs=0, usage=1, timestamp=time.time()),
    ]

    # act
    result = estimator.estimate(schemas)

    # assert
    assert result == 1 / 2


@pytest.mark.unit
def test_estimate_stable_demand_when_inputs_match_outputs(estimator):
    """Tests that a stable demand (1) is estimated when the inputs match the outputs."""
    # arrange
    schemas = [
        PressureSchema(inputs=10, outputs=5, usage=0.1, timestamp=time.time()),
        PressureSchema(inputs=5, outputs=10, usage=0.2, timestamp=time.time()),
        PressureSchema(inputs=2, outputs=2, usage=1, timestamp=time.time()),
    ]

    # act
    result = estimator.estimate(schemas)

    # assert
    assert result == 1
