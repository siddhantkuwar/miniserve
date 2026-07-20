"""
Build this after your attention function works. It should create a tiny input
such as `batch=1, sequence=4, hidden=8, heads=2`, call your implementation,
and print these shapes:

    hidden_states:     [batch, sequence, hidden]
    attention weights: [batch, heads, query_sequence, key_sequence]
    attention values:  [batch, sequence, hidden]

Keep the inputs small enough that you can trace one position by hand.
"""
