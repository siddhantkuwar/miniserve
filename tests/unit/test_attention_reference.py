"""
Keep this module intentionally empty until you have an implementation in
`miniserve.engine.attention`. Add tests in this order:

1. Shape test:
   - use deterministic random inputs;
   - assert attention output is `[batch, sequence, hidden]`;
   - assert attention weights are `[batch, heads, query_sequence, key_sequence]`.

2. Probability test:
   - every row of attention weights should sum to one after softmax.

3. Causality test:
   - run attention once;
   - drastically change only a future token;
   - prove earlier output positions did not change.

4. Validation test:
   - verify an invalid number of heads raises a useful error.

The causality test is the important one. It shows that a language model cannot
read tokens that have not happened yet.
"""
