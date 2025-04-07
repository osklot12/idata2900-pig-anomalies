import pytest

from src.data.dataset.selectors.random_string_selector import RandomStringSelector


@pytest.fixture
def strings():
    """Returns a list of strings."""
    return [f"string_{i}" for i in range(1000)]


@pytest.fixture
def selector(strings):
    """Fixture to provide a RandomStringSelector instance."""
    return RandomStringSelector(strings)


@pytest.mark.unit
def test_next_is_random(selector, strings):
    """Tests that next() returns a pseudo-randomly picked strings."""
    # arrange
    result = []
    runs = 5
    iterations = 20

    # act
    for _ in range(runs):
        picked_files = []
        for _ in range(iterations):
            picked_files.append(selector.next())

        result.append(picked_files)

    # NOTE ABOUT NON-DETERMINISM:
    # with 5 iterations of picking 20 files out of 3 available files,
    # the chance of all lists being equal is ((1/3)^20)^5 = 1/(3^100)
    # which is practically 0. Thus, we can consider a failed test to indicate a true negative.

    # assert
    assert len(set(map(tuple, result))) > 1
