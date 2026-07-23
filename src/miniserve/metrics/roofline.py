"""Assignment 13 scaffold: explicit decode byte model and roofline estimates."""


# TODO: Estimate weight traffic and state whether weights are assumed read once per step.
def weight_bytes_per_step(parameter_count, bits_per_weight, quant_metadata_bytes):
    """Estimate weight traffic and state whether weights are assumed read once per step."""
    pass


# TODO: Estimate K/V reads and writes from layers, KV heads, head dim, and context.
def kv_bytes_per_step(model_config, context_length, batch_size, dtype_bytes):
    """Estimate K/V reads and writes from layers, KV heads, head dim, and context."""
    pass


# TODO: Combine named traffic categories without hiding assumptions.
def total_decode_bytes_per_token(weight_bytes, kv_read_bytes, kv_write_bytes, other_bytes):
    """Combine named traffic categories without hiding assumptions."""
    pass


# TODO: Compute an optimistic memory-bandwidth ceiling.
def bandwidth_limited_tokens_per_second(bandwidth_bytes_per_second, bytes_per_token):
    """Compute an optimistic memory-bandwidth ceiling."""
    pass


# TODO: Report measured implied bandwidth as a fraction of empirical bandwidth.
def achieved_bandwidth_utilization(measured_tokens_per_second, bytes_per_token, bandwidth):
    """Report measured implied bandwidth as a fraction of empirical bandwidth."""
    pass


# TODO: Return FLOPs per byte and explain what region of a roofline it suggests.
def arithmetic_intensity(flops_per_token, bytes_per_token):
    """Return FLOPs per byte and explain what region of a roofline it suggests."""
    pass
