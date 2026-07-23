"""Assignment 16 integration-test prompts for padding and batch correctness."""


# TODO: Use unequal lengths and inspect every mask entry.
def test_padding_mask_marks_only_real_tokens():
    """Use unequal lengths and inspect every mask entry."""
    pass


# TODO: Make padded logits deliberately attractive so wrong indexing fails.
def test_last_valid_logits_ignore_padding_positions():
    """Make padded logits deliberately attractive so wrong indexing fails."""
    pass


# TODO: Compare every request token-by-token under identical prompt formatting.
def test_static_batch_greedy_tokens_match_individual_generation():
    """Compare every request token-by-token under identical prompt formatting."""
    pass


# TODO: Per-request EOS/length state must survive lockstep iterations.
def test_one_finished_row_does_not_stop_other_rows():
    """Per-request EOS/length state must survive lockstep iterations."""
    pass


# TODO: Engine row order must not leak into the API contract.
def test_batch_results_return_in_original_request_order():
    """Engine row order must not leak into the API contract."""
    pass
