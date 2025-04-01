from src.schemas.converters.schema_converter import SchemaConverter
from src.schemas.metric_schema import MetricSchema
from src.schemas.pressure_schema import PressureSchema


class PressureMetricSchemaConverter(SchemaConverter[PressureSchema, MetricSchema]):
    """Converts pressure schemas to metric schemas."""

    def convert(self, schema: PressureSchema) -> MetricSchema:
        return MetricSchema(
            metrics={
                "inputs": schema.inputs,
                "outputs": schema.outputs,
                "usage": schema.usage
            }
        )