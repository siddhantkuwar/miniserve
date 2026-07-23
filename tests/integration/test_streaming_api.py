"""Assignment 15 integration-test prompts using the fake executor first."""


# TODO: Count startup/shutdown calls across the test client lifespan.
def test_lifespan_loads_and_closes_executor_once():
    """Count startup/shutdown calls across the test client lifespan."""
    pass


# TODO: Parse each line independently and verify ordering.
def test_stream_is_valid_jsonl_and_ends_with_one_terminal_event():
    """Parse each line independently and verify ordering."""
    pass


# TODO: Use distinct deterministic token traces to expose cross-talk.
def test_two_concurrent_clients_never_receive_each_others_request_id():
    """Use distinct deterministic token traces to expose cross-talk."""
    pass


# TODO: Abort mid-stream, then inspect executor ownership rather than only HTTP status.
def test_disconnect_cancels_execution_and_releases_state():
    """Abort mid-stream, then inspect executor ownership rather than only HTTP status."""
    pass


# TODO: Boundary validation should reject work before queue admission.
def test_invalid_request_never_reaches_executor():
    """Boundary validation should reject work before queue admission."""
    pass
