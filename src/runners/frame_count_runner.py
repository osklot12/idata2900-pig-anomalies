from src.auth.factories.gcp_auth_service_factory import GCPAuthServiceFactory
from src.data.dataset.label.factories.simple_label_parser_factory import SimpleLabelParserFactory
from src.data.decoders.factories.darwin_decoder_factory import DarwinDecoderFactory
from src.data.loading.loaders.factories.gcs_loader_factory import GCSLoaderFactory
from src.utils.frame_counter import FrameCounter
from src.utils.norsvin_behavior_class import NorsvinBehaviorClass
from src.utils.norsvin_dataset_config import NORSVIN_GCS_CREDS


def main():
    gcs_creds = NORSVIN_GCS_CREDS
    auth_factory = GCPAuthServiceFactory(gcs_creds.service_account_path)
    decoder_factory = DarwinDecoderFactory(
        label_parser_factory=SimpleLabelParserFactory(NorsvinBehaviorClass.get_label_map())
    )
    loader_factory = GCSLoaderFactory(
        bucket_name=gcs_creds.bucket_name,
        auth_factory=auth_factory,
        decoder_factory=decoder_factory
    )

    counter = FrameCounter(loader_factory)

    total_frames = counter.count_total_frames()
    print(f"\n✅ Total frames in dataset: {total_frames}")

    class_video_counts = counter.count_videos_per_class()
    print("\n📦 Number of videos each class appears in:")
    for cls, count in sorted(class_video_counts.items()):
        print(f"  {cls}: {count}")


if __name__ == "__main__":
    main()