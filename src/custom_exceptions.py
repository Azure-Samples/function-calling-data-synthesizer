class InvalidFunctionDefinitionError(Exception):
    """
    Initialize the InvalidFunctionDefinitionError object.
    Custom exception for errors when reading the function definitions.
    Args:
        message (str): The error message.
    """

    def __init__(self, message: str):
        super().__init__(message)


class AllClientsFailedError(Exception):
    """
    Initializes the AllClientsFailedError class.
    Custom exception raised after all GenAI clients failed to respond.
    Args:
        message (str): The error message.
    """

    def __init__(self, message: str):
        super().__init__(message)
