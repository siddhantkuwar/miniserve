"""Assignment 12 scaffold: preallocated contiguous KV storage."""


class PreallocatedKVCache:
    """Own fixed-capacity K/V arrays and append through explicit cursors."""

    # TODO: Allocate storage once and establish canonical `[L,H,T,Dh]` axes.
    def __init__(self, num_layers, num_kv_heads, capacity, head_dim, dtype):
        """Allocate storage once and establish canonical `[L,H,T,Dh]` axes."""
        pass

    # TODO: Associate a request with capacity and a zero cursor.
    def reserve(self, request_id, token_capacity):
        """Associate a request with capacity and a zero cursor."""
        pass

    # TODO: Write into existing storage, enforce bounds, and advance correctly.
    def append(self, request_id, layer, keys, values):
        """Write into existing storage, enforce bounds, and advance correctly."""
        pass

    # TODO: Slice from zero to cursor without exposing uninitialized capacity.
    def view(self, request_id, layer):
        """Slice from zero to cursor without exposing uninitialized capacity."""
        pass

    # TODO: Report the next sequence write position for position-ID reasoning.
    def cursor(self, request_id):
        """Report the next sequence write position for position-ID reasoning."""
        pass

    # TODO: Clear ownership and make capacity reusable according to your design.
    def release(self, request_id):
        """Clear ownership and make capacity reusable according to your design."""
        pass
