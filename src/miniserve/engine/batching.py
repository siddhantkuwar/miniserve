"""Assignment 16 scaffold: static padded prefill and lockstep decode."""


class StaticBatch:
    """Own padded tokens, masks, positions, request mapping, and per-row stop state."""

    pass


# TODO: Select and document padding semantics without confusing pad and EOS.
def choose_pad_token_id(tokenizer):
    """Select and document padding semantics without confusing pad and EOS."""
    pass


# TODO: Build `[batch, max_sequence]` tokens and a validity mask.
def pad_token_sequences(token_sequences, pad_token_id):
    """Build `[batch, max_sequence]` tokens and a validity mask."""
    pass


# TODO: Assign positions only to valid tokens under the chosen left/right padding policy.
def build_position_ids(attention_mask):
    """Assign positions only to valid tokens under the chosen left/right padding policy."""
    pass


# TODO: Select each request's final real-token logits rather than padded positions.
def gather_last_valid_logits(logits, sequence_lengths):
    """Select each request's final real-token logits rather than padded positions."""
    pass


# TODO: Append active rows while preserving finished-row state and request mapping.
def update_batch_with_tokens(batch, next_token_ids, finish_reasons):
    """Append active rows while preserving finished-row state and request mapping."""
    pass


# TODO: Run one fixed group and return per-request traces in original request order.
def static_batch_generate(adapter, requests, config):
    """Run one fixed group and return per-request traces in original request order."""
    pass
