"""Assignment 14 scaffold: one async queue feeding one model executor."""


class FakeExecutor:
    """Emit deterministic fake tokens with controlled delays before using a model."""

    # TODO: Store injected timing behavior for deterministic tests.
    def __init__(self, clock, token_delay):
        """Store injected timing behavior for deterministic tests."""
        pass

    # TODO: Put a validated request into the bounded waiting queue.
    async def submit(self, request):
        """Put a validated request into the bounded waiting queue."""
        pass

    # TODO: Own the execution loop and process requests without mixing output state.
    async def run(self):
        """Own the execution loop and process requests without mixing output state."""
        pass

    # TODO: Transition through prefill/decode and emit fake token events.
    async def process_request(self, request):
        """Transition through prefill/decode and emit fake token events."""
        pass

    # TODO: Mark queued/active work cancelled and guarantee cleanup.
    async def cancel(self, request_id):
        """Mark queued/active work cancelled and guarantee cleanup."""
        pass

    # TODO: Stop accepting work and terminate background tasks cleanly.
    async def shutdown(self):
        """Stop accepting work and terminate background tasks cleanly."""
        pass
