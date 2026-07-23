"""Assignment 11 test prompts for cache invariants and toy parity."""


# TODO: Use asymmetric dimensions so the wrong axis cannot accidentally pass.
def test_append_grows_sequence_axis_by_exact_token_count():
    """Use asymmetric dimensions so the wrong axis cannot accidentally pass."""
    pass


# TODO: Mutating or extending request A must not change request B's view.
def test_request_caches_never_alias():
    """Mutating or extending request A must not change request B's view."""
    pass


# TODO: Every view/append after release should fail with a useful error.
def test_release_removes_all_request_ownership():
    """Every view/append after release should fail with a useful error."""
    pass


# TODO: Compare outputs at every position, not only the final token.
def test_cached_toy_attention_matches_full_recomputation_each_step():
    """Compare outputs at every position, not only the final token."""
    pass


# TODO: Prove your test can catch the bug it claims to guard against.
def test_intentional_position_off_by_one_breaks_parity():
    """Prove your test can catch the bug it claims to guard against."""
    pass
