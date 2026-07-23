"""Assignment 11 scaffold: portable KV-cache semantic contract."""


class KVCache:
    """Describe ownership and append/view/release semantics for one request."""

    # TODO: Claim cache capacity before writes; reject duplicate ownership.
    def reserve(self, request_id, token_capacity):
        """Claim cache capacity before writes; reject duplicate ownership."""
        pass

    # TODO: Append matching K/V token slices and advance the request cursor once.
    def append(self, request_id, layer, keys, values):
        """Append matching K/V token slices and advance the request cursor once."""
        pass

    # TODO: Return only initialized tokens in the canonical axis order.
    def view(self, request_id, layer):
        """Return only initialized tokens in the canonical axis order."""
        pass

    # TODO: Remove ownership and make future access fail clearly.
    def release(self, request_id):
        """Remove ownership and make future access fail clearly."""
        pass

    # TODO: Return logical bytes currently owned for observability.
    def bytes_used(self, request_id):
        """Return logical bytes currently owned for observability."""
        pass
