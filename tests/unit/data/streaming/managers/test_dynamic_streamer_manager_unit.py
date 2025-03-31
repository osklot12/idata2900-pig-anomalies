from unittest.mock import MagicMock, create_autospec

import pytest
import time

from src.data.streaming.factories.streamer_factory import StreamerFactory
from src.data.streaming.managers.dynamic_streamer_manager import DynamicStreamerManager
from src.schemas.pressure_schema import PressureSchema


@pytest.fixture
def demand_estimator():
    """Fixture to provide a DemandEstimator instance."""
    estimator = MagicMock()
    estimator.estimate.return_value = 2.0
    return estimator


def make_streamer_factory(sleep_duration: float = 0.2):
    """Creates a streamer factory with streamers that sleep during wait_for_completion."""
    factory = MagicMock()

    def create_streamer():
        streamer = MagicMock()
        streamer.wait_for_completion.side_effect = lambda: time.sleep(sleep_duration)
        streamer.stop_streaming.return_value = None
        return streamer

    factory.create_streamer.side_effect = create_streamer
    return factory


def make_manager(estimator, factory, min_s=1, max_s=5):
    """Creates a DynamicStreamerManager instance."""
    return DynamicStreamerManager(
        streamer_factory=factory,
        min_streamers=min_s,
        max_streamers=max_s,
        demand_estimator=estimator,
        stability=10
    )


@pytest.mark.unit
def test_max_streamers_launched_on_run(demand_estimator):
    """Tests that the max amount of streamers are launched initially when ran."""
    # arrange
    manager = make_manager(demand_estimator, make_streamer_factory())

    # act
    manager.run()
    time.sleep(.1)

    # assert
    assert manager.n_active_streamers() == 5

    manager.stop()


@pytest.mark.unit
def test_n_streamers_adjust_with_pressure(demand_estimator):
    """Tests that the number of streamers running are adjusted with pressure reports."""
    # arrange
    demand_estimator.estimate.return_value = 0.1
    manager = make_manager(demand_estimator, make_streamer_factory())

    manager.run()
    time.sleep(.1)

    manager.new_schema(PressureSchema(0, 0, 0, 0))  # lower demand first
    time.sleep(.2)
    first_n = manager.n_active_streamers()

    # act
    demand_estimator.estimate.return_value = 2
    manager.new_schema(PressureSchema(0, 0, 0, 0))
    time.sleep(.2)

    second_n = manager.n_active_streamers()

    # assert
    assert first_n < 5
    assert second_n > first_n

    manager.stop()