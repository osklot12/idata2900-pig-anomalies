import time
from collections import deque
from typing import Deque, Tuple

from src.schemas.schemas.metric_schema import MetricSchema
from src.schemas.schemas.pressure_schema import PressureSchema
from src.schemas.schemas.schema_listener import SchemaListener


class PressureToMetric(SchemaListener[PressureSchema]):
    """Aggregator for aggregating pressure schema data into metric schemas."""

    def __init__(self, consumer: SchemaListener[MetricSchema], stability: int = 100):
        """
        Initializes a PressureToMetric instance.

        Args:
            consumer (SchemaListener[MetricSchema]): the consumer to forward the metric schema to
            stability (int): the number of pressure schemas to aggregate for
        """
        if stability < 1:
            raise ValueError("stability cannot be less than 1")

        self._consumer = consumer
        self._queue: Deque[PressureSchema] = deque(maxlen=stability)

    def new_schema(self, schema: PressureSchema) -> None:
        self._queue.append(schema)
        self._consumer.new_schema(self._create_metric_schema())

    def _create_metric_schema(self) -> MetricSchema:
        """Creates a metric schema from the pressure schemas."""
        inputs, outputs = self._compute_avg_in_out()
        return MetricSchema(
            metrics={
                "inputs/s": inputs,
                "outputs/s": outputs,
                "usage": self._queue[-1].usage
            },
            timestamp = time.time()
        )

    def _compute_avg_in_out(self) -> Tuple[float, float]:
        """Computes the average number of inputs and outputs each second."""
        inputs = 0
        outputs = 0

        first = self._queue[0]
        min_timestamp = first.timestamp
        max_timestamp = first.timestamp
        for s in self._queue:
            if s.timestamp < min_timestamp:
                min_timestamp = s.timestamp

            if s.timestamp > max_timestamp:
                max_timestamp = s.timestamp

            inputs += s.inputs
            outputs += s.outputs

        interval = max_timestamp - min_timestamp
        if interval <= 0:
            avg_inputs = 0.0
            avg_outputs = 0.0
        else:
            avg_inputs = inputs / interval
            avg_outputs = outputs / interval

        return avg_inputs, avg_outputs
