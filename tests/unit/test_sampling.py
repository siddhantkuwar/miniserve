"""Assignment 8 test prompts for probability and stopping semantics."""


# TODO: Compare entropy or probability gaps on a tiny logit vector.
def test_temperature_below_one_sharpens_distribution():
    """Compare entropy or probability gaps on a tiny logit vector."""
    pass


# TODO: Choose distinct logits so ties do not obscure the invariant.
def test_top_k_masks_exactly_vocabulary_size_minus_k_entries():
    """Choose distinct logits so ties do not obscure the invariant."""
    pass


# TODO: Protect the non-empty nucleus edge case.
def test_top_p_keeps_first_token_even_when_it_exceeds_threshold():
    """Protect the non-empty nucleus edge case."""
    pass


# TODO: Check finite values and declared tolerance.
def test_probabilities_sum_to_one_after_filtering():
    """Check finite values and declared tolerance."""
    pass


# TODO: Reproducibility requires owning and recording random state.
def test_seeded_sampling_replays_the_same_sequence():
    """Reproducibility requires owning and recording random state."""
    pass


# TODO: Lifecycle metrics need the reason, not only that generation ended.
def test_stop_reason_distinguishes_eos_from_length_limit():
    """Lifecycle metrics need the reason, not only that generation ended."""
    pass
