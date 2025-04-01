import pytest
import time

from src.schemas.aggregators.pressure_to_metric import PressureToMetric
from src.schemas.schemas.metric_schema import MetricSchema
from src.schemas.schemas.pressure_schema import PressureSchema
from src.schemas.schemas.schema_listener import SchemaListener


class CapturingConsumer(SchemaListener[MetricSchema]):
    """A schema consumer for testing."""

    def __init__(self):
        self.metrics = []

    def new_schema(self, schema: MetricSchema) -> None:
        self.metrics.append(schema)


@pytest.fixture
def consumer():
    """Fixture to provide a consumer of metric schemas."""
    return CapturingConsumer()


@pytest.fixture
def aggregator(consumer):
    """Fixture to provide a PressureToMetric instance."""
    return PressureToMetric(consumer, stability=10)


@pytest.mark.unit
def test_throughput_over_interval(aggregator, consumer):
    """Tests that throughput over the interval is correctly computed."""
    # arrange
    base_time = time.time()
    schemas = [
        PressureSchema(inputs=10, outputs=5, usage=0.3, timestamp=base_time),
        PressureSchema(inputs=20, outputs=10, usage=0.5, timestamp=base_time + 1),
        PressureSchema(inputs=30, outputs=15, usage=0.7, timestamp=base_time + 2)
    ]

    # act
    for schema in schemas:
        aggregator.new_schema(schema)

    # assert
    metric = consumer.metrics[-1]
    print(metric)
    assert metric.metrics["inputs/s"] == pytest.approx((10 + 20 + 30) / 2)
    assert metric.metrics["outputs/s"] == pytest.approx((5 + 10 + 15) / 2)
    assert metric.metrics["usage"] == 0.7
