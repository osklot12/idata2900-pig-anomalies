import sys
import threading
import time
from typing import Tuple, List

from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from src.data.structures.atomic_bool import AtomicBool
from src.schemas.metric_schema import MetricSchema
from src.schemas.observer.schema_listener import SchemaListener
from src.schemas.signed_schema import SignedSchema


class RichDashboard(SchemaListener[SignedSchema[MetricSchema]]):
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

        self._render_thread = None

    def new_schema(self, schema: SignedSchema[MetricSchema]) -> None:
        self._schemas[schema.issuer_id] = schema

    def _render_tables(self) -> Columns:
        schemas = list(self._schemas.values())

        grouped: dict[Tuple[str, ...], List[SignedSchema[MetricSchema]]] = {}
        for s in schemas:
            key_set = tuple(sorted(s.schema.metrics.keys()))
            grouped.setdefault(key_set, []).append(s)

        tables = []

        for metric_keys, group in grouped.items():
            title = " | ".join(metric_keys)
            table = Table(title=f"Metrics: {title}")
            table.add_column("Component ID")
            for key in metric_keys:
                table.add_column(key, justify="right", width=10, no_wrap=True)
            table.add_column("Last updated")

            for schema in sorted(group, key=lambda sch: sch.issuer_id):
                row = [schema.issuer_id]
                for key in metric_keys:
                    value = schema.schema.metrics.get(key, "-")
                    row.append(str(value))
                row.append(time.strftime("%H:%M:%S", time.localtime(schema.schema.timestamp)))
                table.add_row(*row)

            tables.append(Panel(table))

        return Columns(tables)

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
        with Live(self._render_tables(), refresh_per_second=int(1 / self._interval), console=self.console) as live:
            while self._running:
                live.update(self._render_tables())
                time.sleep(self._interval)

    def stop(self) -> None:
        """Stops the dashboard."""
        self._running.set(False)
        self._render_thread.join()
        self._render_thread = None