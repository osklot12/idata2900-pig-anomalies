from abc import ABC, abstractmethod

class ConfigInterface(ABC):
    """Defines the interface for configuration management."""

    @abstractmethod
    def get(self, key: str, default=None):
        """
        Retrieves a configuration value by key.
        :param key: A string key for the config value (e.g., 'augmentation.rotation_range').
        :param default: The default value to return if the key is not found.
        :return: The config value or the default.
        """
        pass
