class AudioProcessingException(Exception):
    """Exception for errors occurring while processing audio."""
    pass


class AudioProducingException(Exception):
    """Exception for errors occurring while producing audio."""
    pass


class MessageSendException(Exception):
    """Exception for errors occurring while sending messages."""
    pass


class MessageReceiveException(Exception):
    """Exception for errors occurring while receiving messages."""
    pass


class FileHandlingException(Exception):
    """Exception for errors occurring while handling files."""
    pass


class SingletonException(Exception):
    """Exception for errors related to Singleton instances."""
    pass


class SetupException(Exception):
    """Exception for errors occurring during the system's setup phase."""
    pass


class FinishedSongException(Exception):
    """Exception raised when a recorded song's file is finished (the song is over)."""
    pass
