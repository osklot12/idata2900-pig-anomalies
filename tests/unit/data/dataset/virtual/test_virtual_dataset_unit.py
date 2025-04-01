from typing import Tuple

import pytest

from src.data.dataclasses.identifiable import Identifiable
from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.splitters.consistent_dataset_splitter import ConsistentDatasetSplitter
from src.data.dataset.virtual.virtual_dataset import VirtualDataset, O, I
from src.schemas.observer.schema_broker import SchemaBroker
from src.schemas.observer.schema_listener import SchemaListener, T
from src.schemas.observer.schema_signer_broker import SchemaSignerBroker
from src.schemas.pressure_schema import PressureSchema
from src.schemas.signed_schema import SignedSchema


class FakeFood(Identifiable):
    """A concrete identifiable for testing."""

    def __init__(self, id_: str, value: int) -> None:
        self._id = id_
        self._value = value

    def get_id(self) -> str:
        return self._id

    def get_value(self) -> int:
        return self._value


class DummyDataset(VirtualDataset[FakeFood, FakeFood]):
    """A concrete VirtualDataset for testing with simple FakeFood instances."""

    def _get_identified_instance(self, food: I) -> Tuple[str, O]:
        return food.get_id(), food.get_value()

    def _frame_in_instance(self, instance: O) -> int:
        return 1


class DummyComponentListener(SchemaListener[SignedSchema[PressureSchema]]):
    """A concrete component listener for pressure schemas."""

    def __init__(self):
        self.schemas = []

    def new_schema(self, schema: SignedSchema[PressureSchema]) -> None:
        print(f"Got schema {schema} from {schema.issuer_id}")
        self.schemas.append(schema)


@pytest.fixture
def splitter():
    """Fixture to provide a DatasetSplitter instance."""
    return ConsistentDatasetSplitter(train_ratio=1, val_ratio=0)


@pytest.fixture
def dataset(splitter):
    """Fixture to provide a VirtualDataset instance."""
    return DummyDataset(splitter, max_size=6)


@pytest.mark.unit
def test_feed_and_len(dataset):
    """Tests that feeding the VirtualDataset will increase its 'length'"""
    # act
    dataset.feed(FakeFood("a", 1))
    dataset.feed(FakeFood("b", 2))
    dataset.feed(FakeFood("c", 3))

    length = len(dataset)

    # assert
    assert length == 3


@pytest.mark.unit
def test_split_assignment_consistency(splitter):
    """Tests that the same food will always end up in the same split."""
    # arrange
    dataset = DummyDataset(splitter, max_size=1)
    item = FakeFood("abc", 42)

    # act
    dataset.feed(item)
    split_before = splitter.get_split_for_id(item.get_id())

    dataset.feed(FakeFood("evict", -1))  # Evicts 'abc'

    dataset.feed(item)
    split_after = splitter.get_split_for_id(item.get_id())

    # assert
    assert split_before == split_after


@pytest.mark.unit
def test_get_batch_returns_correct_size(dataset):
    """Tests that get_batch() returns a batch with the expected size."""
    # arrange
    batch_size = 5
    for i in range(batch_size):
        dataset.feed(FakeFood(f"id_{i}", i))

    # act
    batch = dataset.get_batch(DatasetSplit.TRAIN, batch_size)

    # assert
    assert len(batch) == 5


@pytest.mark.unit
def test_split_size_return_correct_split_size(dataset):
    """Tests that split_size() returns the correct split size."""
    # arrange
    n = 4
    for i in range(n):
        dataset.feed(FakeFood(f"id_{i}", i))

    # act
    train_size = dataset.split_size(DatasetSplit.TRAIN)
    val_size = dataset.split_size(DatasetSplit.VAL)
    test_size = dataset.split_size(DatasetSplit.TEST)

    # assert
    assert train_size == n
    assert val_size == test_size == 0


@pytest.mark.unit
def test_feed_same_id_does_not_duplicate(dataset):
    """Tests that feeding with same ID overwrites instead of adding a duplicate."""
    # arrange
    first_item = FakeFood("a", 1)
    second_item = FakeFood("a", 2)

    # act
    dataset.feed(first_item)
    dataset.feed(second_item)

    # assert
    assert len(dataset) == 1
    assert dataset._buffer_map.get(DatasetSplit.TRAIN).at(first_item.get_id()) == second_item.get_value()


@pytest.mark.unit
def test_eviction_happens_in_correct_order(dataset):
    """Tests that eviction of instances happens in a FIFO manner."""
    # act
    items = []
    for i in range(8):
        item = FakeFood(f"id_{i}", i)
        items.append(item)
        dataset.feed(item)

    # assert
    buffer = dataset._buffer_map.get(DatasetSplit.TRAIN)
    keys = set(buffer.keys())

    assert "id_0" not in keys
    assert "id_1" not in keys

    for i in range(2, 8):
        assert f"id_{i}" in keys


@pytest.mark.unit
def test_get_batch_raises_on_too_large_request(dataset):
    """Tests that get_batch() raises when the requested batch size is too large."""
    dataset.feed(FakeFood("x", 1))
    with pytest.raises(ValueError):
        dataset.get_batch(DatasetSplit.TRAIN, 2)


@pytest.mark.unit
def test_pressure_reporting(splitter):
    """Tests that pressure is reported correctly when a ComponentEventBroker is provided."""
    # arrange
    broker = SchemaSignerBroker[PressureSchema]("test-dataset")
    listener = DummyComponentListener()
    broker.get_schema_broker().subscribe(listener)
    dataset = DummyDataset(splitter, max_size=10, pressure_broker=broker)

    # act
    for i in range(3):
        dataset.feed(FakeFood(f"id_{i}", i))

    dataset.get_batch(DatasetSplit.TRAIN, 3)

    # assert
    schemas = listener.schemas
    assert len(schemas) == 4

    last_size = 0
    for i in range(3):
        assert schemas[i].schema.inputs == 1
        assert schemas[i].schema.outputs == 0
        assert schemas[i].schema.usage > last_size
        last_size = schemas[i].schema.usage

    assert schemas[3].schema.inputs == 0
    assert schemas[3].schema.outputs == 3
    assert schemas[3].schema.usage == last_size
