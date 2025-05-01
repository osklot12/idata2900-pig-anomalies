from src.auth.factories.gcp_auth_service_factory import GCPAuthServiceFactory
from src.data.dataset.label.factories.simple_label_parser_factory import SimpleLabelParserFactory
from src.data.dataset.metamakers.file_metamaker import FileMetamaker
from src.data.dataset.splitters.factories.string_set_splitter_factory import StringSetSplitterFactory
from src.data.decoders.factories.darwin_decoder_factory import DarwinDecoderFactory
from src.data.loading.loaders.factories.gcs_loader_factory import GCSLoaderFactory
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from src.utils.norsvin_dataset_config import NORSVIN_GCS_CREDS


def main():
    gcs_creds = NORSVIN_GCS_CREDS
    auth_factory = GCPAuthServiceFactory(service_account_file=gcs_creds.service_account_path)
    decoder_factory = DarwinDecoderFactory(
        label_parser_factory=SimpleLabelParserFactory(NorsvinBehaviorClass.get_label_map())
    )
    loader_factory = GCSLoaderFactory(
        bucket_name=gcs_creds.bucket_name,
        auth_factory=auth_factory,
        decoder_factory=decoder_factory
    )
    maker = FileMetamaker(loader_factory, splitter_factory=StringSetSplitterFactory(weights=[0.8, 0.1, 0.1]), cache=True)
    metadata = maker.make_metadata()
    print(metadata)


if __name__ == "__main__":
    main()