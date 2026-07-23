"""Assignment 11 scaffold: correctness-first concatenating KV cache."""


class NaiveConcatKVCache:
    """Grow arrays by concatenation to provide a simple correctness oracle."""

    # TODO: Create empty per-request, per-layer K/V ownership tables.
    def __init__(self, num_layers):
        """Create empty per-request, per-layer K/V ownership tables."""
        pass

    # TODO: Record ownership; capacity is metadata because this oracle grows dynamically.
    def reserve(self, request_id, token_capacity):
        """Record ownership; capacity is metadata because this oracle grows dynamically."""
        pass

    # TODO: Concatenate one or more new token positions in canonical axis order.
    def append(self, request_id, layer, keys, values):
        """Concatenate one or more new token positions in canonical axis order."""
        pass

    # TODO: Return all cached keys/values for the request and layer.
    def view(self, request_id, layer):
        """Return all cached keys/values for the request and layer."""
        pass

    # TODO: Delete every per-layer array owned by the request.
    def release(self, request_id):
        """Delete every per-layer array owned by the request."""
        pass
