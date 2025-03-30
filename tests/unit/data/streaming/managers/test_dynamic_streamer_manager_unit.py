from unittest.mock import MagicMock

import pytest
import time

from src.data.streaming.factories.streamer_factory import StreamerFactory
from src.data.streaming.managers.dynamic_streamer_manager import DynamicStreamerManager
from src.data.streaming.streamers.streamer import Streamer
from src.schemas.algorithms.demand_estimator import DemandEstimator
from src.schemas.pressure_schema import PressureSchema


@pytest.fixture
def factory():
    """Fixture to provide a StreamerFactory instance."""
    factory = MagicMock(spec=StreamerFactory)

    def make_streamer():
        streamer = MagicMock(spec=Streamer)
        # Simulate actual work by sleeping
        streamer.wait_for_completion.side_effect = lambda: time.sleep(.1)
        return streamer

    factory.create_streamer.side_effect = make_streamer
    return factory


@pytest.fixture
def estimator():
    """Fixture to provide a DemandEstimator instance."""
    return MagicMock(spec=DemandEstimator)


@pytest.mark.unit
def test_demand_affects_growth(factory, estimator):
    """Tests that the estimated demand affects the amount of streamers maintained."""
    # arrange
    estimator.estimate.return_value = .5
    manager = DynamicStreamerManager(
        streamer_factory=factory,
        min_streamers=0,
        max_streamers=100,
        demand_estimator=estimator
    )
    manager.run()
    time.sleep(0.2)
    n_streamers_before = manager.n_active_streamers()
    schema = PressureSchema(inputs=0, outputs=0, occupied=0, timestamp=time.time())

    # act
    for i in range(100):
        manager.new_schema(schema)
    time.sleep(0.2)

    n_streamers_after = manager.n_active_streamers()
    manager.stop()

    # assert
    assert n_streamers_before > n_streamers_after
