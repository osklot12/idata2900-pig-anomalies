import pytest
import numpy as np
from src.data.structures.hash_buffer import HashBuffer


@pytest.mark.unit
def test_hash_buffer_add():
    """Tests adding elements to the buffer."""
    # arrange
    buffer = HashBuffer[int, int](max_size=3)

    # act
    buffer.add(1, 100)
    buffer.add(2, 200)
    buffer.add(3, 300)

    # assert
    assert len(buffer) == 3
    assert buffer.has(1)
    assert buffer.has(2)
    assert buffer.has(3)


@pytest.mark.unit
def test_hash_buffer_pop():
    """Tests popping elements from the buffer."""
    # arrange
    buffer = HashBuffer[int, int](max_size=3)
    first_added_element = 100
    second_added_element = 200
    third_added_element = 300
    buffer.add(1, first_added_element)
    buffer.add(2, second_added_element)
    buffer.add(3, third_added_element)

    # act
    first_popped_element = buffer.pop(1)
    second_popped_element = buffer.pop(2)
    third_popped_element = buffer.pop(3)

    # assert
    assert first_popped_element == first_added_element
    assert second_popped_element == second_added_element
    assert third_popped_element == third_added_element

    assert len(buffer) == 0
    assert buffer.pop(0) is None


@pytest.mark.unit
def test_hash_buffer_eviction():
    """Tests that the buffer evicts old entries when full."""
    # arrange
    buffer = HashBuffer[int, int](max_size=3)
    buffer.add(1, 100)
    buffer.add(2, 200)
    buffer.add(3, 300)

    # act
    evicted_keys = buffer.add(4, 400)

    # assert
    assert not buffer.has(1)
    assert buffer.has(2)
    assert buffer.has(3)
    assert buffer.has(4)

    assert evicted_keys == [1]


@pytest.mark.unit
def test_hash_buffer_thread_safety():
    """Tests concurrent access to the buffer."""
    # arrange
    import threading
    buffer = HashBuffer[int, int](max_size=5)

    def add_item():
        for i_ in range(10):
            buffer.add(i_, i_ * 10)

    # act
    threads = [threading.Thread(target=add_item) for _ in range(3)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # assert
    assert len(buffer) == 5
    for i in range(5, 10):
        assert buffer.has(i)


@pytest.mark.unit
def test_hash_buffer_remove_nonexistent():
    """Tests that popping a non-existent index returns None."""
    # arrange
    buffer = HashBuffer[int, int](max_size=2)

    # act
    popped_element = buffer.pop(17)

    # assert
    assert popped_element is None
