"""Errors raised by the remote client."""

from typing import Optional


class RemoteError(RuntimeError):
    """An exception raised by the hosted object (or the server) while handling a call.

    ``type`` is the remote exception's class name and ``remote_traceback`` its
    formatted traceback; exception objects themselves never cross the wire.
    """

    def __init__(self, type: str, message: str, remote_traceback: Optional[str] = None) -> None:
        super().__init__(f"{type}: {message}")
        self.type, self.message, self.remote_traceback = type, message, remote_traceback or ""

    def __str__(self) -> str:
        text = super().__str__()
        return f"{text}\n\nRemote traceback:\n{self.remote_traceback}" if self.remote_traceback else text
