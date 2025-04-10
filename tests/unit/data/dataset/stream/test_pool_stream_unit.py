from unittest.mock import Mock

import pytest

from src.data.dataset.streams.pool_stream import PoolStream


@pytest.fixture
def augmented():
    """Fixture to provide an 'augmented' string."""
    return "augmented"


@pytest.fixture
def augmentor(augmented):
    """Fixture to provide an Augmentor instance."""
    augmentor = Mock()
    augmentor.augment.return_value = [augmented]
    return augmentor


@pytest.mark.unit
def test_feed_no_augmentor():
    """Tests that feeding data without an augmentor should preserve the data."""
    # arrange
    stream = PoolStream(pool_size=1, augmentor=None)
    s = "string"

    # act
    stream.feed(s)

    instance = stream.read()

    # assert
    assert instance == s


@pytest.mark.unit
def test_feed_with_augmentor(augmentor, augmented):
    """Tests that feeding data with an augmentor augments the data."""
    # arrange
    stream = PoolStream(pool_size=1, augmentor=augmentor)
    s = "string"

    # act
    stream.feed(s)

    instance = stream.read()

    assert instance == augmented
