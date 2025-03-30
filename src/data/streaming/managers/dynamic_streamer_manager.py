from collections import deque
from threading import Lock
from typing import Deque

from src.data.streaming.factories.streamer_factory import StreamerFactory
from src.data.streaming.managers.concurrent_streamer_manager import ConcurrentStreamerManager
from src.data.streaming.streamers.streamer import Streamer
from src.schemas.observer.schema_listener import SchemaListener, T
from src.schemas.pressure_schema import PressureSchema


class DynamicStreamerManager(ConcurrentStreamerManager, SchemaListener[PressureSchema]):
    """A streamer manager that adjust the number of concurrent streamers dynamically."""

    def __init__(self, streamer_factory: StreamerFactory, min_streamers: int, max_streamers: int):
        """
        Initializes a DynamicStreamerManager instance.

        Args:
            streamer_factory (StreamerFactory): the factory used to create the streamers
            min_streamers (int): the minimum number of concurrent streamers
            max_streamers (int): the maximum number of concurrent streamers
        """
        if min_streamers < 1:
            raise ValueError("min_streamers must be greater than 0")

        if min_streamers > max_streamers:
            raise ValueError("min_streamers must be less than or equal to max_streamers")

        super().__init__(max_streamers)
        self._min_streamers = min_streamers
        self._factory = streamer_factory

        self.__pressure_points: Deque[PressureSchema] = deque(maxlen=100)

        self.__target_n_streamers = 0
        self.__compute_lock = Lock()

    def _setup(self) -> None:
        for _ in range(self._min_streamers):
            self._launch_streamer(self._factory.create_streamer())

    def _run_streamer(self, streamer: Streamer, streamer_id: str) -> None:
        streamer.wait_for_completion()
        streamer.stop_streaming()

    def _handle_done_streamer(self, streamer_id: str) -> None:
        self._remove_streamer(streamer_id)
        self._launch_optimal_n_streamers()

    def _handle_crashed_streamer(self, streamer_id: str, e: Exception) -> None:
        print(f"Streamer {streamer_id} crashed with exception {e}")
        self._remove_streamer(streamer_id)
        self._launch_optimal_n_streamers()

    def new_schema(self, component_id: str, schema: T) -> None:
        with self.__compute_lock:
            self.__pressure_points.append(schema)
            self.__set_optimal_streamers()
        self._launch_optimal_n_streamers()

    def _launch_optimal_n_streamers(self) -> None:
        """Launches streamers to reach the optimal number of concurrent streamers."""
        while self.n_active_streamers() < self._get_target_n_streamers():
            self._launch_streamer(self._factory.create_streamer())

    def _get_target_n_streamers(self) -> int:
        """Returns the number of current streamers."""
        with self.__compute_lock:
            return self.__target_n_streamers

    def __set_optimal_streamers(self) -> None:
        """Sets the optimal number of streamers to maintain."""
        optimal_n = self.__compute_optimal_streamers()
        self.__target_n_streamers = max(self._min_streamers, min(optimal_n, self._max_streamers))

    def __compute_optimal_streamers(self) -> int:
        """Computes the optimal number of streamers to maintain."""
        if not self.__pressure_points:
            raise ValueError("Cannot compute optimal number of streamers, no pressure points available")

        internal_pressure = self.__pressure_points[-1].occupied
        if internal_pressure < 1:
            optimal_n = self._max_streamers
        else:
            optimal_n = self.__get_optimal_streamers_ext()

        return optimal_n

    def __get_optimal_streamers_ext(self):
        """Computes the optimal number of streamers based on external pressure for the receiver."""
        external_pressure = self.__get_external_pressure()

        if external_pressure == 0:
            optimal_n = self.__target_n_streamers
        elif external_pressure < 0:
            optimal_n = self.__target_n_streamers + 1
        else:
            optimal_n = self.__target_n_streamers - 1

        return optimal_n

    def __get_external_pressure(self) -> float:
        """Computes the external pressure put on the receiver."""
        if not self.__pressure_points:
            raise ValueError("Cannot compute pressure, no pressure points available")

        numerator = 0
        denominator = 0
        for point in self.__pressure_points:
            numerator += point.inputs - point.outputs
            denominator += point.inputs + point.outputs

        if denominator == 0:
            result = 0
        else:
            result = numerator / denominator

        return result
