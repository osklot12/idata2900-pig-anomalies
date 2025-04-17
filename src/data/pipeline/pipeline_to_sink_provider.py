from typing import TypeVar, Optional

from src.data.pipeline.consumer import Consumer
from src.data.pipeline.consumer_provider import ConsumerProvider
from src.data.structures.atomic_bool import AtomicBool

T = TypeVar("T")

class PipelineProvider(ConsumerProvider[T]):
    """Provider of complete pipelines."""



    def get_consumer(self, release: Optional[AtomicBool] = None) -> Optional[Consumer[T]]:
        pass