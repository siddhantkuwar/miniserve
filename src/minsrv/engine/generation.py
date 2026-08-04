"""Assignment 7 scaffold: MiniServe's uncached manual greedy decoder."""


# TODO: Select `[batch, vocabulary]` logits from the final sequence position.
def select_last_position_logits(logits):
    """Select `[batch, vocabulary]` logits from the final sequence position."""
    pass


# TODO: Choose the maximum-logit token ID without using a generation helper.
def greedy_next_token(logits):
    """Choose the maximum-logit token ID without using a generation helper."""
    pass


# TODO: Create the next full-sequence input by appending one token on sequence axis.
def append_token(token_ids, next_token_id):
    """Create the next full-sequence input by appending one token on sequence axis."""
    pass


# TODO: Yield one structured step after each full-sequence model forward pass.
def uncached_generate_steps(adapter, prompt_token_ids, max_new_tokens):
    """Yield one structured step after each full-sequence model forward pass."""
    pass


# TODO: Own encode-loop-stop-decode control flow and return text plus token trace.
def generate_text(adapter, prompt, max_new_tokens):
    """Own encode-loop-stop-decode control flow and return text plus token trace."""
    pass


# TODO: Parse CLI arguments and run one deterministic prompt through MiniServe.
def main():
    """Parse CLI arguments and run one deterministic prompt through MiniServe."""
    pass
