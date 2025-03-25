class DatasetFile:
    """Represents a file in the dataset."""

    def __init__(self, file_path: str, instance_id: str):
        """
        Initializes a DatasetFile instance.

        Args:
            file_path (str): the path to the file
            instance_id (str): the ID of the dataset instance
        """
        self._file_path = file_path
        self._instance_id = instance_id

    def get_instance_id(self) -> str:
        """
        Returns the instance ID.

        Returns:
            str: the instance ID
        """
        return self._instance_id
