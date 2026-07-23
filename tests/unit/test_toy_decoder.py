"""Assignment 4 test prompts for explicit autoregressive control flow."""


# TODO: Use a tiny literal vector and avoid random data for this first test.
def test_greedy_selection_returns_highest_logit_index():
    """Use a tiny literal vector and avoid random data for this first test."""
    pass


# TODO: Inspect the trace length and each sequence prefix.
def test_generation_appends_exactly_one_token_per_step():
    """Inspect the trace length and each sequence prefix."""
    pass


# TODO: Ensure EOS ends before max_new_tokens when encountered first.
def test_generation_stops_on_eos():
    """Ensure EOS ends before max_new_tokens when encountered first."""
    pass


# TODO: Protect against accidental infinite loops.
def test_generation_stops_at_max_new_tokens_without_eos():
    """Protect against accidental infinite loops."""
    pass


# TODO: Establish determinism as the future golden-token foundation.
def test_same_input_produces_same_greedy_trace():
    """Establish determinism as the future golden-token foundation."""
    pass
