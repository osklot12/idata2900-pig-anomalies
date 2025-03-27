import multiprocessing
from abc import ABC, abstractmethod

from src.data.dataclasses.annotated_frame import AnnotatedFrame


class DataProvider(ABC):

    @abstractmethod
    def get_queue(self) -> multiprocessing.Queue:
        raise NotImplementedError