"""Assignment 4 scaffold: tiny deterministic autoregressive decoding.

Use a tiny vocabulary and a deterministic next-logit function. The goal is to
own the control flow before a pretrained model adds architecture complexity.
"""


# TODO: Return next-token logits for each input position from a toy transition table.
def toy_forward(token_ids, transition_logits):
    """Return next-token logits for each input position from a toy transition table."""
    pass


# TODO: Return the token ID with the largest score along the vocabulary axis.
def select_greedy_token(next_token_logits):
    """Return the token ID with the largest score along the vocabulary axis."""
    pass


# TODO: Decide whether EOS or the explicit generation budget ended decoding.
def should_stop(token_id, eos_token_id, generated_count, max_new_tokens):
    """Decide whether EOS or the explicit generation budget ended decoding."""
    pass


# TODO: Append exactly one selected token per loop iteration and return the trace.
def generate_toy(token_ids, transition_logits, eos_token_id, max_new_tokens):
    """Append exactly one selected token per loop iteration and return the trace."""
    pass


# TODO: Run one deterministic toy prompt and print each decoding step.
def main():
    """Run one deterministic toy prompt and print each decoding step."""
    pass
