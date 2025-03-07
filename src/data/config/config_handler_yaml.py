import yaml
from src.config.config_interface import ConfigInterface

class YamlConfigHandler(ConfigInterface):
    """Loads configuration from a YAML file."""

    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)

    def _load_config(self, path):
        """Load YAML configuration file."""
        try:
            with open(path, "r") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Warning: Config file '{path}' not found. Using defaults.")
            return {}

    def get(self, key, default=None):
        """
        Retrieves a configuration value by key.
        Supports nested keys using dot notation (e.g., "augmentation.rotation_range").
        """
        keys = key.split(".")
        value = self.config
        for k in keys:
            value = value.get(k, default) if isinstance(value, dict) else default
        return value
