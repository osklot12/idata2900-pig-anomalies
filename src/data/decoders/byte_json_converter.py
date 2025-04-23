import json

from src.data.decoders.json_converter import JSONConverter, T


class ByteJSONConverter(JSONConverter[bytes]):
    """Reads JSON files from bytes."""

    def get_json(self, data: bytes) -> dict:
        return json.loads(data.decode("utf-8"))
