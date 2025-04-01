from abc import ABC

from src.schemas.metric_schema import MetricSchema
from src.schemas.observer.schema_listener import SchemaListener, T
from src.schemas.signed_schema import SignedSchema


class ComponentMonitor(SchemaListener[SignedSchema[MetricSchema]], ABC):
    """Interface for component telemetry."""