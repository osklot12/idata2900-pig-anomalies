from collections import deque
from threading import Lock
from typing import Deque
import math

from src.data.streaming.factories.streamer_factory import StreamerFactory
from src.data.streaming.managers.concurrent_streamer_manager import ConcurrentStreamerManager
from src.data.streaming.streamers.streamer import Streamer
from src.schemas.algorithms.demand_estimator import DemandEstimator
from src.schemas.observer.schema_listener import SchemaListener, T
from src.schemas.pressure_schema import PressureSchema


class DynamicStreamerManager(ConcurrentStreamerManager, SchemaListener[PressureSchema]):
    """A streamer manager that adjust the number of concurrent streamers dynamically."""

    def __init__(self, streamer_factory: StreamerFactory, min_streamers: int, max_streamers: int,
                 demand_estimator: DemandEstimator):
        """
        Initializes a DynamicStreamerManager instance.

        Args:
            streamer_factory (StreamerFactory): the factory used to create the streamers
            min_streamers (int): the minimum number of concurrent streamers
            max_streamers (int): the maximum number of concurrent streamers
            demand_estimator (DemandEstimator): the demand estimator used to estimate the streamer demand
        """
        if min_streamers < 0:
            raise ValueError("min_streamers cannot be negative")

        if min_streamers > max_streamers:
            raise ValueError("min_streamers must be less than or equal to max_streamers")

        super().__init__(max_streamers)
        self._min_streamers = min_streamers
        self._factory = streamer_factory

        self._demand_estimator = demand_estimator

        self.__pressure_schemas: Deque[PressureSchema] = deque(maxlen=100)
        self.__optimal_n_streamers = float(self._max_streamers)
        self.__lock = Lock()
        self.__schema_lock = Lock()

    def _setup(self) -> None:
        self._manage_streamers()

    def _run_streamer(self, streamer: Streamer, streamer_id: str) -> None:
        streamer.wait_for_completion()
        streamer.stop_streaming()

    def _handle_done_streamer(self, streamer_id: str) -> None:
        self._remove_streamer(streamer_id)
        self._executor.submit(self._manage_streamers)

    def _handle_crashed_streamer(self, streamer_id: str, e: Exception) -> None:
        print(f"Streamer {streamer_id} crashed with exception {e}")
        self._remove_streamer(streamer_id)
        self._executor.submit(self._manage_streamers)

    def __set_optimal_n_streamers(self, n: float) -> None:
        """Sets the optimal number of concurrent streamers."""
        with self.__lock:
            self.__optimal_n_streamers = min(max(float(self._min_streamers), n), float(self._max_streamers))

    def _manage_streamers(self) -> None:
        with self.__lock:
            n_to_launch = max(0, round(self.__optimal_n_streamers) - self.n_active_streamers())
            for _ in range(n_to_launch):
                self._launch_streamer(self._factory.create_streamer())

    def new_schema(self, schema: T) -> None:
        with self.__schema_lock:
            self.__pressure_schemas.append(schema)
            self.__set_optimal_n_streamers(self.__compute_optimal_n_streamers())
        self._executor.submit(self._manage_streamers)

    def __compute_optimal_n_streamers(self) -> float:
        """Computes the optimal number of concurrent streamers."""
        demand = self._demand_estimator.estimate(list(self.__pressure_schemas))
        delta = math.log(max(demand, 1e-6))
        return self.__optimal_n_streamers + delta