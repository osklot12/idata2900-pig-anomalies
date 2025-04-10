from src.data.dataset.providers.dataset_entity_provider import DatasetEntityProvider
from src.data.dataset.providers.factories.entity_provider_factory import EntityProviderFactory
from src.data.dataset.providers.lazy_entity_provider import LazyEntityProvider
from src.data.loading.factories.loader_factory import LoaderFactory
from src.data.parsing.factories.string_parser_factory import StringParserFactory


class LazyEntityProviderFactory(EntityProviderFactory):
    """Factory for creating LazyEntityProvider instances."""

    def __init__(self, loader_factory: LoaderFactory, id_parser_factory: StringParserFactory):
        """
        Initializes a LazyEntityProviderFactory instance.

        Args:
            loader_factory (LoaderFactory): factory for creating loaders
            id_parser_factory (StringParserFactory): factory for creating id parsers
        """
        self._loader_factory = loader_factory
        self._id_parser_factory = id_parser_factory

    def create_provider(self) -> DatasetEntityProvider:
        return LazyEntityProvider(
            loader_factory=self._loader_factory,
            id_parser=self._id_parser_factory.create_string_parser()
        )