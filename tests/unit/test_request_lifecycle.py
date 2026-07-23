"""Assignment 14 test prompts for lifecycle and async ownership."""


# TODO: Table-drive valid and invalid edges from the PRD state diagram.
def test_only_declared_state_transitions_are_allowed():
    """Table-drive valid and invalid edges from the PRD state diagram."""
    pass


# TODO: Use an injected fake clock rather than sleeping.
def test_timestamps_are_monotonic_and_first_token_is_written_once():
    """Use an injected fake clock rather than sleeping."""
    pass


# TODO: Admission and cache allocation must remain separate operations.
def test_waiting_request_owns_no_cache():
    """Admission and cache allocation must remain separate operations."""
    pass


# TODO: Run concurrent consumers and detect any queue/state cross-talk.
def test_two_requests_receive_only_their_own_fake_tokens():
    """Run concurrent consumers and detect any queue/state cross-talk."""
    pass


# TODO: Assert cleanup for waiting and active cancellation separately.
def test_cancellation_releases_capacity_and_closes_stream():
    """Assert cleanup for waiting and active cancellation separately."""
    pass
