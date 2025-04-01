import threading
import time
from typing import Tuple, List, Dict, Type

from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from typing_extensions import TypeVar

from src.data.structures.atomic_bool import AtomicBool
from src.schemas.converters.schema_converter import SchemaConverter
from src.schemas.schemas.metric_schema import MetricSchema
from src.schemas.schemas.schema_listener import SchemaListener
from src.schemas.schemas.schema import Schema
from src.schemas.schemas.signed_schema import SignedSchema

S = TypeVar("S", bound=Schema)


class RichDashboard(SchemaListener[SignedSchema[Schema]]):
    """A Rich telemetry dashboard."""

    def __init__(self, refresh_interval: float = 0.5) -> None:
        """
        Initializes a RichDashboard instance.

        Args:
            refresh_interval (float): the refresh interval in seconds
        """
        self._schemas: dict[str, SignedSchema[MetricSchema]] = {}
        self._interval = refresh_interval
        self._running = AtomicBool(False)
        self.console = Console()

        self._converter_registry: Dict[Type[Schema], SchemaConverter[Schema, MetricSchema]] = {}
        self._max_logs = 30

        self._render_thread = None

    def new_schema(self, schema: SignedSchema) -> None:
        schema_type = type(schema.schema)
        converter = self._converter_registry.get(schema_type)

        if converter is None:
            raise KeyError(f"Schema type {schema_type} is not registered with any converter")

        metric_schema = converter.convert(schema.schema)
        self._schemas[schema.signature] = SignedSchema(
            signature=schema.signature,
            timestamp=schema.timestamp,
            schema=metric_schema
        )

    def register(self, input_type: Type[S], converter: SchemaConverter[S, MetricSchema]) -> None:
        """
        Registers a schema converter.

        Args:
            input_type (Type[Schema]): the type of schema to use the converter for
            converter (SchemaConverter[Schema, SignedSchema[MetricSchema]]): the converter instance
        """
        self._converter_registry[input_type] = converter

    def get_converter(self, input_type: Type[Schema]) -> SchemaConverter[Schema, MetricSchema]:
        """
        Returns the converter for the given schema type.

        Args:
            input_type (Type[Schema]): the schema type

        Returns:
            SchemaConverter[Schema, SignedSchema[MetricSchema]]: the converter instance
        """
        return self._converter_registry.get(input_type, None)

    def _render_tables(self) -> Columns:
        """Renders the tables."""
        grouped = self._group_schemas_by_attributes()
        tables = [self._build_table(metric_keys, group) for metric_keys, group in grouped.items()]
        return Columns([Panel(table) for table in tables])

    @staticmethod
    def _build_table(metric_keys: Tuple[str, ...], group: List[SignedSchema[MetricSchema]]) -> Table:
        """Builds a table for a set of metrics and metric schemas."""
        title = " | ".join(metric_keys)
        table = Table(title=f"Metrics: {title}")
        table.add_column("Component ID")
        for key in metric_keys:
            table.add_column(key, justify="right", width=10, no_wrap=True)
        table.add_column("Last updated")

        for schema in sorted(group, key=lambda sch: sch.signature):
            table.add_row(*RichDashboard._build_row(schema, metric_keys))

        return table

    @staticmethod
    def _build_row(schema: SignedSchema[MetricSchema], metric_keys: Tuple[str, ...]) -> List[str]:
        """Builds a row for a set of metric keys from a metric schema."""
        row = [Text(schema.signature, style="italic yellow")]

        for key in metric_keys:
            value = schema.schema.metrics.get(key, "-")
            value = f"{value:.3f}"
            row.append(value)
        row.append(time.strftime("%H:%M:%S", time.localtime(schema.timestamp)))

        return row

    def _group_schemas_by_attributes(self) -> Dict[Tuple[str, ...], List[SignedSchema[MetricSchema]]]:
        """Groups the stored schemas by attributes."""
        grouped: dict[Tuple[str, ...], List[SignedSchema[MetricSchema]]] = {}

        schemas = list(self._schemas.values())
        for s in schemas:
            key_set = tuple(sorted(s.schema.metrics.keys()))
            grouped.setdefault(key_set, []).append(s)

        return grouped

    def start(self, blocking: bool = False) -> None:
        """
        Starts the dashboard.

        Args:
            blocking (bool): whether the rendering is blocking or not
        """
        self._running.set(True)

        if blocking:
            self._render_loop()
        else:
            self._render_thread = threading.Thread(target=self._render_loop, daemon=True)
            self._render_thread.start()

    def _render_loop(self) -> None:
        """The render loop to render the dashboard continuously."""
        with Live(self._render_tables(), refresh_per_second=int(1 / self._interval), console=self.console) as live:
            while self._running:
                live.update(self._render_tables())
                time.sleep(self._interval)

    def stop(self) -> None:
        """Stops the dashboard."""
        self._running.set(False)
        self._render_thread.join()
        self._render_thread = None
