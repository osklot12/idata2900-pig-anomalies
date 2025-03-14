from abc import ABC, abstractmethod

from src.data.pipeline.pipeline_configurator import PipelineConfigurator


class PipelineConfigProvider(ABC):
    """An interface for factories of pipeline configurators."""

    @abstractmethod
    def get_configurator(self) -> PipelineConfigurator:
        """
        Returns a pipeline configurator instance.

        Returns:
            PipelineConfigurator: A pipeline configurator instance.
        """
        raise NotImplementedError