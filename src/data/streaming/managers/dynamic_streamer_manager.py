import threading
from collections import deque
from threading import Lock
from typing import Deque
import math
import time

from src.data.streaming.streamers.providers.streamer_factory import StreamerFactory
from src.data.streaming.managers.concurrent_streamer_manager import ConcurrentStreamerManager
from src.data.streaming.streamers.streamer import Streamer
from src.data.structures.atomic_bool import AtomicBool
from src.data.structures.atomic_var import AtomicVar
from src.schemas.algorithms.demand_estimator import DemandEstimator
from src.schemas.schemas.schema_listener import SchemaListener
from src.schemas.schemas.pressure_schema import PressureSchema


class DynamicStreamerManager(ConcurrentStreamerManager, SchemaListener[PressureSchema]):
    """A streamer manager that adjust the number of concurrent streamers dynamically."""

    def __init__(self, streamer_factory: StreamerFactory, min_streamers: int, max_streamers: int,
                 demand_estimator: DemandEstimator, stability: int = 100):
        """
        Initializes a DynamicStreamerManager instance.

        Args:
            streamer_factory (StreamerFactory): the factory used to create the streamers
            min_streamers (int): the minimum number of concurrent streamers
            max_streamers (int): the maximum number of concurrent streamers
            demand_estimator (DemandEstimator): the demand estimator used to estimate the streamer demand
            stability (int): the number of previous pressure schemas used for demand estimation
        """
        if min_streamers < 0:
            raise ValueError("min_streamers cannot be negative")

        if min_streamers > max_streamers:
            raise ValueError("min_streamers must be less than or equal to max_streamers")

        if stability < 1:
            raise ValueError("stability cannot be less than 1")

        super().__init__(max_streamers)
        self._min_streamers = min_streamers
        self._streamer_factory = streamer_factory

        self._demand_estimator = demand_estimator

        self._pressure_schemas: Deque[PressureSchema] = deque(maxlen=stability)
        self._optimal_n_streamers = AtomicVar[float](float(self._max_streamers))
        self._lock = Lock()
        self._schema_lock = Lock()

        self._controller_thread = None
        self._controller_running = AtomicBool(False)
        self._controller_lock = Lock()

    def _setup(self) -> None:
        self._controller_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._controller_running.set(True)
        self._controller_thread.start()

    def _run_streamer(self, streamer: Streamer, streamer_id: str) -> None:
        streamer.wait_for_completion()
        streamer.stop_streaming()

    def _handle_done_streamer(self, streamer_id: str) -> None:
        self._remove_streamer(streamer_id)

    def _handle_crashed_streamer(self, streamer_id: str, e: Exception) -> None:
        self._remove_streamer(streamer_id)

    def _control_loop(self):
        """Loop for controlling the number of concurrent streamers."""
        while self._controller_running:
            self._adjust_streamers()
            time.sleep(.1)

    def _adjust_streamers(self):
        """Adjusts the number of currently running streamers to the optimal number."""
        with self._controller_lock:
            n_to_launch = max(0, round(self._optimal_n_streamers.get()) - self.n_active_streamers())
            for _ in range(n_to_launch):
                self._launch_streamer(self._streamer_factory.create_streamer())

    def new_schema(self, schema: PressureSchema) -> None:
        with self._schema_lock:
            self._pressure_schemas.append(schema)
            self._optimal_n_streamers.set(self._compute_optimal_n_streamers())

    def _compute_optimal_n_streamers(self) -> float:
        """Computes the optimal number of concurrent streamers."""
        demand = self._demand_estimator.estimate(list(self._pressure_schemas))

        baseline = self._optimal_n_streamers.get()
        step = math.log2(max(demand, 1e-3))

        scaled = baseline + step
        return max(float(self._min_streamers), min(float(self._max_streamers), scaled))

    def _stop(self) -> None:
        self._controller_running.set(False)
        self._controller_thread.join()
        self._controller_thread = None
