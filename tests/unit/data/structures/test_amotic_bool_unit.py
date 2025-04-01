import threading

import pytest

from src.data.structures.atomic_bool import AtomicBool


@pytest.fixture
def boolean():
    """Fixture to provide an AtomicBool instance."""
    return AtomicBool()


@pytest.mark.unit
def test_default_value_is_false(boolean):
    """Tests that the default value of AtomicBool is false."""
    # act
    value = boolean.get()

    # assert
    assert value is False


@pytest.mark.unit
def test_set_and_get(boolean):
    """Tests that get and set works as expected."""
    # act
    boolean.set(True)
    first_value = boolean.get()
    first_bool_value = bool(boolean)

    boolean.set(False)
    second_value = boolean.get()
    second_bool_value = bool(boolean)

    # assert
    assert first_value == first_bool_value == True
    assert second_value == second_bool_value == False


@pytest.mark.unit
def test_toggle(boolean):
    """Tests that toggling works as expected."""
    # act
    first_value = boolean.toggle()
    second_value = boolean.toggle()
    third_value = boolean.toggle()

    # assert
    assert first_value == third_value == True
    assert second_value == False


@pytest.mark.unit
def test_thread_safety_set_get(boolean):
    """Tests that get and set are thread-safe."""
    # arrange
    def set_true():
        for _ in range(1000):
            boolean.set(True)

    def set_false():
        for _ in range(1000):
            boolean.set(False)

    threads = [threading.Thread(target=set_true), threading.Thread(target=set_false)]

    # act
    [t.start() for t in threads]
    [t.join() for t in threads]

    # assert
    assert boolean.get() in (True, False)


@pytest.mark.unit
def test_thread_safety_toggle(boolean):
    """Tests that toggle is thread-safe."""
    # arrange
    def toggle_n(n):
        for _ in range(n):
            boolean.toggle()

    threads = [threading.Thread(target=toggle_n, args=(1000,)) for _ in range(4)]

    # act
    [t.start() for t in threads]
    [t.join() for t in threads]

    # assert
    assert boolean.get() is False
