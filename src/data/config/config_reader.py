import json

class ConfigReader:
    @staticmethod
    def read(path: str) -> dict:
        with open(path) as f:
            return json.load(f)
