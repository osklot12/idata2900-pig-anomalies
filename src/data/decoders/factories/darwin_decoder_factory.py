from src.data.decoders.annotation_decoder import AnnotationDecoder
from src.data.decoders.darwin_decoder import DarwinDecoder
from src.data.decoders.factories.annotation_decoder_factory import AnnotationDecoderFactory
from src.data.dataset.label.factories.label_parser_factory import LabelParserFactory


class DarwinDecoderFactory(AnnotationDecoderFactory):
    """A factory for creating DarwinDecoder instances."""

    def __init__(self, label_parser_factory: LabelParserFactory):
        """
        Initializes a DarwinDecoderFactory instance.

        Args:
            label_parser_factory (LabelParserFactory): a factory that creates label parsers
        """
        self._label_parser_factory = label_parser_factory

    def create_decoder(self) -> AnnotationDecoder:
        return DarwinDecoder(self._label_parser_factory.create_label_parser())