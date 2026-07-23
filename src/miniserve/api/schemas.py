"""Assignment 15 scaffold: HTTP boundary schemas, separate from engine state."""


class CompletionRequest:
    """Validate prompt, max tokens, sampling settings, and optional client ID."""

    pass


class TokenEvent:
    """Represent one streamed token with request ID, index, text, and timestamp."""

    pass


class CompletionEvent:
    """Represent terminal finish reason, usage counts, and lifecycle metrics."""

    pass


class ErrorEvent:
    """Represent a stable machine-readable public error without leaking internals."""

    pass
