"""Assignment 2 test prompts. Write every assertion yourself."""


# TODO: Use a hand-checkable tensor and verify normalization over hidden only.
def test_layer_norm_has_zero_mean_and_unit_variance_before_affine_terms():
    """Use a hand-checkable tensor and verify normalization over hidden only."""
    pass


# TODO: Verify batch/sequence survive expansion and projection unchanged.
def test_feed_forward_preserves_outer_shape():
    """Verify batch/sequence survive expansion and projection unchanged."""
    pass


# TODO: Prove residual connections preserve input when sublayers return zero.
def test_zero_sublayers_reduce_block_to_identity():
    """Prove residual connections preserve input when sublayers return zero."""
    pass


# TODO: Carry Assignment 1's causal invariant through the whole block.
def test_future_token_cannot_change_past_block_outputs():
    """Carry Assignment 1's causal invariant through the whole block."""
    pass
