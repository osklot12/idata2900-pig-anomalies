from abc import ABC

from src.schemas.technical.metric_schema import MetricSchema
from src.schemas.schema_listener import SchemaListener
from src.schemas.signed_schema import SignedSchema


class ComponentMonitor(SchemaListener[SignedSchema[MetricSchema]], ABC):
    """Interface for component telemetry."""