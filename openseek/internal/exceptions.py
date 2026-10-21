class OpenSeekException(BaseException):
    pass


class MissingCredentials(OpenSeekException):
    pass


class InvalidCredentials(OpenSeekException):
    pass


class ServerDown(OpenSeekException):
    pass


class RateLimited(ServerDown):
    """An explicit upstream refusal, not a broken browser or partial answer."""

    retry_after = 10


class MissingInitialization(OpenSeekException):
    pass


class VerificationRequired(OpenSeekException):
    pass


class FileUploadError(OpenSeekException):
    pass
