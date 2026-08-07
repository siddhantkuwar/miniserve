"""Assignment 7 scaffold: MiniServe's uncached manual greedy decoder."""
import mlx.core as core

# TODO: Select `[batch, vocabulary]` logits from the final sequence position.
def select_last_position_logits(logits):
    """Select `[batch, vocabulary]` logits from the final sequence position."""
    logit = logits[:, -1, :]
    
    if (logits.ndim != 3):
        raise ValueError(f"needed shape [B, T, V] but got {logits.shape} instead")
    
    return logit


# TODO: Choose the maximum-logit token ID without using a generation helper.
def greedy_next_token(logits):
    """Choose the maximum-logit token ID without using a generation helper."""
    # what the, greedy decoding is just argmax(logits) but for the V axis
    
    return core.argmax(logits, axis=-1)


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
    input = "What is the capital of France?"
    
    

if __name__ == "__main__":
    main()
