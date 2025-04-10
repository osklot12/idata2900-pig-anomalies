from src.data.dataset.manifests.factories.manifest_factory import ManifestFactory
from src.data.dataset.providers.factories.instance_provider_factory import InstanceProviderFactory
from src.data.dataset.providers.instance_provider import InstanceProvider
from src.data.dataset.providers.manifest_instance_provider import ManifestInstanceProvider
from src.data.dataset.selectors.factories.string_selector_factory import StringSelectorFactory


class ManifestInstanceProviderFactory(InstanceProviderFactory):
    """Factory for creating ManifestInstanceProvider instances."""

    def __init__(self, manifest_factory: ManifestFactory, selector_factory: StringSelectorFactory):
        """
        Initializes a ManifestInstanceProviderFactory instance.

        Args:
            manifest_factory (ManifestFactory): factory for creating file manifests
            selector_factory (StringSelectorFactory): factory for creating file selectors
        """
        self._manifest_factory = manifest_factory
        self._selector_factory = selector_factory

    def create_provider(self) -> InstanceProvider:
        return ManifestInstanceProvider(
            manifest=self._manifest_factory.create_manifest(),
            selector=self._selector_factory.create_selector()
        )