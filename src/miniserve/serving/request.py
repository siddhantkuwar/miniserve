"""Assignment 14 scaffold: request state, timestamps, and lifecycle invariants."""


class RequestState:
    """Define CREATED, WAITING, PREFILL, DECODING, FINISHED, CANCELLED, FAILED."""

    pass


class Request:
    """Own prompt/generated tokens, limits, timestamps, output queue, and finish reason."""

    # TODO: Create a request in CREATED without allocating model cache.
    def __init__(self, request_id, prompt_tokens, max_new_tokens, arrival_ns):
        """Create a request in CREATED without allocating model cache."""
        pass

    # TODO: Validate allowed state edges and record monotonic lifecycle times.
    def transition(self, next_state, timestamp_ns):
        """Validate allowed state edges and record monotonic lifecycle times."""
        pass

    # TODO: Record one generated token and first-token time exactly once.
    def append_token(self, token_id, timestamp_ns):
        """Record one generated token and first-token time exactly once."""
        pass

    # TODO: Enter FINISHED once and make completion idempotence explicit.
    def finish(self, reason, timestamp_ns):
        """Enter FINISHED once and make completion idempotence explicit."""
        pass

    # TODO: Enter CANCELLED from any cancellable state and trigger cleanup ownership.
    def cancel(self, timestamp_ns):
        """Enter CANCELLED from any cancellable state and trigger cleanup ownership."""
        pass

    # TODO: Derive queue delay, TTFT, generation duration, and end-to-end latency.
    def metrics(self):
        """Derive queue delay, TTFT, generation duration, and end-to-end latency."""
        pass
