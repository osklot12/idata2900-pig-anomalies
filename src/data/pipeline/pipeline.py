from src.data.pipeline.pipline_config_provider import PipelineConfigProvider


class Pipeline:
    """A facade for full data pipelines."""

    def __init__(self, config_provider: PipelineConfigProvider):
        """
        Creates a new pipeline instance.

        Args:
            config_provider (PipelineConfigProvider): a pipeline config provider.
        """
        self.config = config_provider.get_configurator()
        self.virtual_dataset = self.config.config_virtual_dataset()
        self.instance_loader = self.config.config_instance_loader()

    def run(self):
        """Runs the pipeline."""

    def stop(self):
        """Stops the pipeline."""