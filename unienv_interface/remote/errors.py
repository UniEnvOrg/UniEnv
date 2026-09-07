"""Stable remote failure categories; exception objects never cross the wire."""


class RemoteError(RuntimeError):
    def __init__(self, code: str, message: str, *, uncertain: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.uncertain = uncertain


class UncertainOutcomeError(RemoteError):
    def __init__(self, message: str = "Connection lost or timed out; the remote mutation may have executed") -> None:
        super().__init__("uncertain_outcome", message, uncertain=True)
