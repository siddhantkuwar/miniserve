"""Assignment 12 scaffold: separate prefill from one-token cached decode."""


# TODO: Process the full prompt once, populate cache, and return final-position logits.
def prefill(adapter, prompt_token_ids, cache, request_id):
    """Process the full prompt once, populate cache, and return final-position logits."""
    pass


# TODO: Process exactly one new token while reading prior keys/values from cache.
def decode_one(adapter, token_id, cache, request_id, position):
    """Process exactly one new token while reading prior keys/values from cache."""
    pass


# TODO: Run one prefill followed by one-token decode iterations with explicit positions.
def cached_generate_steps(adapter, prompt_token_ids, cache, request_id, max_new_tokens):
    """Run one prefill followed by one-token decode iterations with explicit positions."""
    pass


# TODO: Derive layer × K/V × heads × tokens × head_dim × bytes.
def estimate_request_cache_bytes(model_config, token_capacity, dtype_bytes):
    """Derive layer × K/V × heads × tokens × head_dim × bytes."""
    pass
