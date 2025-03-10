import pytest
import threading

from src.data.data_structures.concurrent_dict import ConcurrentDict


@pytest.fixture
def concurrent_dict():
    """Fixture that provides a new ConcurrentDict instance for each test."""
    return ConcurrentDict[str, int]()

def test_set_and_get(concurrent_dict):
    """Tests that setting and getting values works correctly."""
    # act
    concurrent_dict.set("a", 1)

    # assert
    assert concurrent_dict.get("a") == 1

def test_remove(concurrent_dict):
    """Tests that removing a key deletes it from the dictionary."""
    # arrange
    concurrent_dict.set("key1", 10)

    # act
    concurrent_dict.remove("key1")

    # assert
    assert not concurrent_dict.contains("key1")

def test_contains(concurrent_dict):
    """Tests that contains() correctly checks for key existence."""
    # arrange
    concurrent_dict.set("key1", 10)

    # act
    exists = concurrent_dict.contains("key1")
    nonexistent = concurrent_dict.contains("nonexistent")

    # assert
    assert exists
    assert not nonexistent

def test_items(concurrent_dict):
    """Tests that items() returns all key-value pairs in a thread-safe copy."""
    # arrange
    concurrent_dict.set("one", 1)
    concurrent_dict.set("two", 2)

    # act
    items = concurrent_dict.items()

    # assert
    assert set(items) == {("one", 1), ("two", 2)}

def test_keys(concurrent_dict):
    """Tests that keys() returns all keys in a thread-safe copy."""
    # arrange
    concurrent_dict.set("one", 1)
    concurrent_dict.set("two", 2)

    # act
    keys = concurrent_dict.keys()

    # assert
    assert set(keys) == {"one", "two"}

def test_values(concurrent_dict):
    """Tests that values() returns all values in a thread-safe copy."""
    # arrange
    concurrent_dict.set("one", 1)
    concurrent_dict.set("two", 2)

    # act
    values = concurrent_dict.values()

    # assert
    assert set(values) == {1, 2}

def test_len(concurrent_dict):
    """Tests that __len__ correctly returns the number of items."""
    # arrange
    initial_len = len(concurrent_dict)
    concurrent_dict.set("one", 1)
    concurrent_dict.set("two", 2)

    # act
    final_len = len(concurrent_dict)

    # assert
    assert initial_len == 0
    assert final_len == 2

def test_thread_safety(concurrent_dict):
    """Tests that the ConcurrentDict remains consistent under concurrent access."""
    # arrange
    def writer():
        for i in range(1000):
            concurrent_dict.set(f"key-{i}", i)

    def remover():
        for i in range(500):
            concurrent_dict.remove(f"key-{i}")

    threads = [threading.Thread(target=writer) for _ in range(5)] + [threading.Thread(target=remover) for _ in range(3)]

    # act
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # assert
    keys = concurrent_dict.keys()
    assert all(key.startswith("key-") for key in keys)
    assert len(concurrent_dict) == 500