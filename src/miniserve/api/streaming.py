"""Assignment 15 scaffold: turn request events into a cancellable JSONL stream."""


# TODO: Serialize exactly one event followed by one newline.
def encode_jsonl_event(event):
    """Serialize exactly one event followed by one newline."""
    pass


# TODO: Yield events, detect disconnect, cancel work, and always release resources.
async def stream_request_events(request_state, http_request, executor):
    """Yield events, detect disconnect, cancel work, and always release resources."""
    pass


# TODO: Provide a narrow, testable boundary around transport disconnect detection.
async def wait_for_disconnect(http_request):
    """Provide a narrow, testable boundary around transport disconnect detection."""
    pass
