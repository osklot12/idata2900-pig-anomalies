from abc import ABC, abstractmethod


class ModelTrainer(ABC):
    """
    Generic interface for model training.
    """

    @abstractmethod
    def train(self) -> str:
        """
        Executes training loop of model.

        Returns:
            str: A summary or final result message.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self) -> str:
        """
        Executes evaluation loop of model.

        Returns:
            str: A summary or final result message.
        """
        raise NotImplementedError