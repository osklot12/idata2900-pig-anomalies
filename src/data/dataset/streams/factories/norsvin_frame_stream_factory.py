from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataset.streams.factories.stream_factory import StreamFactory
from src.data.dataset.streams.stream import Stream


class NorsvinFrameStreamFactory(StreamFactory[AnnotatedFrame]):
    """Dataset streams factory for the Norsvin dataset."""

    def create_training_stream(self) -> Stream[AnnotatedFrame]:
        pass

    def create_validation_stream(self) -> Stream[AnnotatedFrame]:
        pass

    def create_test_stream(self) -> Stream[AnnotatedFrame]:
        pass