class DataRetrievalError(Exception):
    """Raised when an error related to data retrieval occurs."""

    def __init__(self, message: str, cause: Exception = None):
        """
        Initializes a DataRetrievalError instance.

        Args:
            message (str): the error message
            cause (Exception): the causing exception
        """
        super().__init__(message)
        self.__cause__ = cause