"""Assignment 7 unit-test prompts using a fake adapter."""


# TODO: Make earlier positions disagree so an indexing bug is visible.
def test_only_final_sequence_position_selects_next_token():
    """Make earlier positions disagree so an indexing bug is visible."""
    pass


# TODO: Use a fake adapter call counter to expose control flow.
def test_generation_calls_forward_once_per_generated_token():
    """Use a fake adapter call counter to expose control flow."""
    pass


# TODO: Record input shapes: prompt, prompt+1, prompt+2, and so on.
def test_each_uncached_forward_receives_the_full_growing_sequence():
    """Record input shapes: prompt, prompt+1, prompt+2, and so on."""
    pass


# TODO: Define whether EOS is returned in token trace and keep it consistent.
def test_eos_token_is_included_once_then_generation_stops():
    """Define whether EOS is returned in token trace and keep it consistent."""
    pass


# TODO: Enforce the project's from-scratch generation boundary.
def test_runtime_path_never_calls_mlx_lm_generate():
    """Enforce the project's from-scratch generation boundary."""
    pass
