import json
from pathlib import Path

from src.auth.factories.gcp_auth_service_factory import GCPAuthServiceFactory
from src.data.dataset.label.factories.simple_label_parser_factory import SimpleLabelParserFactory
from src.data.dataset.metamakers.file_metamaker import FileMetamaker
from src.data.decoders.factories.darwin_decoder_factory import DarwinDecoderFactory
from src.data.loading.loaders.factories.gcs_loader_factory import GCSLoaderFactory
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from src.utils.norsvin_dataset_config import NORSVIN_GCS_CREDS
from src.utils.path_finder import PathFinder


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
    maker = FileMetamaker(loader_factory)
    metadata = maker.make_metadata()

    output_path = PathFinder.get_abs_path("metadata_output/metadata.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_serializable = {
        str(k): {str(label): count for label, count in labels.items()}
        for k, labels in metadata.items()
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata_serializable, f, indent=2)

    print(f"Metadata written to {output_path}")


if __name__ == "__main__":
    main()