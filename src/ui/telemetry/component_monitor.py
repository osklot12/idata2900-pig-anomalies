from abc import ABC

from src.schemas.schemas.metric_schema import MetricSchema
from src.schemas.schemas.schema_listener import SchemaListener
from src.schemas.schemas.signed_schema import SignedSchema


class ComponentMonitor(SchemaListener[SignedSchema[MetricSchema]], ABC):
    """Interface for component telemetry."""