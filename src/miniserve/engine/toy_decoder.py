"""Assignment 4 scaffold: tiny deterministic autoregressive decoding.

Use a tiny vocabulary and a deterministic next-logit function. The goal is to
own the control flow before a pretrained model adds architecture complexity.
"""
import torch

# TODO: Return next-token logits for each input position from a toy transition table.
def toy_forward(token_ids, transition_logits):
    """Return next-token logits for each input position from a toy transition table."""
    token_id_tensor = torch.tensor(token_ids, dtype=torch.long)
    all_logits = transition_logits[token_id_tensor]
    
    print("token ID tensor shape:", token_id_tensor.shape)
    print("transition table shape:", transition_logits.shape)
    print("all logits shape:", all_logits.shape)
    
    return all_logits


# TODO: Return the token ID with the largest score along the vocabulary axis.
def select_greedy_token(next_token_logits):
    """Return the token ID with the largest score along the vocabulary axis."""
    winning_index = torch.argmax(next_token_logits)
    token_id = winning_index.item()
    
    print("winning index tensor:", winning_index)
    print("winning index type:", type(winning_index))
    print("token ID:", token_id)
    print("token ID type:", type(token_id))
    
    return token_id


# TODO: Decide whether EOS or the explicit generation budget ended decoding.
def should_stop(token_id, eos_token_id, generated_count, max_new_tokens):
    """Decide whether EOS or the explicit generation budget ended decoding."""
    if (token_id == eos_token_id or generated_count >= max_new_tokens):
        return True
    
    return False


# TODO: Append exactly one selected token per loop iteration and return the trace.
def generate_toy(token_ids, transition_logits, eos_token_id, max_new_tokens):
    """Append exactly one selected token per loop iteration and return the trace."""
    pass


# TODO: Run one deterministic toy prompt and print each decoding step.
def main():
    """Run one deterministic toy prompt and print each decoding step."""
    token_ids = [0, 1, 2, 3, 4, 5, 6]
    V = 9
    eos_token_id = 8
    
    transition_logits = torch.full((V, V), -10.0)
    transition_logits[0, 1] = 10.0  # What -> is
    transition_logits[1, 2] = 10.0  # is -> the
    transition_logits[2, 3] = 10.0  # the -> capital
    transition_logits[3, 4] = 10.0  # capital -> of
    transition_logits[4, 5] = 10.0  # of -> France
    transition_logits[5, 6] = 10.0  # France -> ?
    transition_logits[6, 7] = 10.0  # ? -> Paris
    transition_logits[7, 8] = 10.0  # Paris -> EOS
    transition_logits[8, 8] = 10.0  # EOS -> EOS
    
    #print("row after '?': ", transition_logits[6])
    #print("winning next ID: ", torch.argmax(transition_logits[6]))
    
    all_logits = toy_forward(token_ids, transition_logits)

    next_token_logits = all_logits[-1]
    print("next token logits shape:", next_token_logits.shape)

    next_token_id = select_greedy_token(next_token_logits)
    print("selected next token:", next_token_id)
    
    #stop = should_stop(next_token_id, eos_token_id, generated_count, max_new_tokens)
    
    #testing the stop func
    result = should_stop(
        token_id=7,
        eos_token_id=8,
        generated_count=1,
        max_new_tokens=5,
    )

    print("stop func test result: ", result)
    
    

if __name__ == "__main__":
    main()