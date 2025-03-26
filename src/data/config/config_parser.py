from src.data.config.pipeline_config import PipelineConfig

class ConfigParser:
    @staticmethod
    def parse(cfg_dict: dict) -> PipelineConfig:
        return PipelineConfig(**cfg_dict)
