import torch


def toy_forward(token_ids, transition_logits):
    token_id_tensor = torch.tensor(token_ids, dtype=torch.long)
    all_logits = transition_logits[token_id_tensor]

    """
    print("token ID tensor shape:", token_id_tensor.shape)
    print("transition table shape:", transition_logits.shape)
    print("all logits shape:", all_logits.shape)
    """

    return all_logits


def select_greedy_token(next_token_logits):
    winning_index = torch.argmax(next_token_logits)
    token_id = winning_index.item()

    """
    print("winning index tensor:", winning_index)
    print("winning index type:", type(winning_index))
    print("token ID:", token_id)
    print("token ID type:", type(token_id))
    """

    return token_id


def should_stop(token_id, eos_token_id, generated_count, max_new_tokens):
    if token_id == eos_token_id or generated_count >= max_new_tokens:
        return True

    return False


def generate_toy(token_ids, transition_logits, eos_token_id, max_new_tokens):
    current_tokens = token_ids.copy()
    generated_count = 0
    trace = []

    """
    print("initial tokens:", current_tokens)
    print("generated count:", generated_count)
    print("trace:", trace)
    """

    while True:
        all_logits = toy_forward(current_tokens, transition_logits)
        # print("all logits shape:", all_logits.shape)

        final_row = all_logits[-1]
        # print("final row: ", final_row)

        greedy = select_greedy_token(final_row)
        # print("greedy token selection: ", greedy)

        tokens_before = current_tokens.copy()

        current_tokens.append(greedy)
        # print("after appending: ", current_tokens)

        generated_count += 1

        stop = should_stop(greedy, eos_token_id, generated_count, max_new_tokens)

        step = {
            "step": generated_count,
            "tokens_before": tokens_before,
            "selected_token_id": greedy,
            "tokens_after": current_tokens.copy(),
            "generated_count": generated_count,
            "stop": stop,
        }

        trace.append(step)

        if stop:
            break

    return current_tokens, trace


def render_tokens(token_ids, id_to_token):
    return " ".join(id_to_token[token_id] for token_id in token_ids)


def main():
    token_ids = [0, 1, 2, 3, 4, 5, 6]
    V = 9
    eos_token_id = 8

    id_to_token = {
        0: "What",
        1: "is",
        2: "the",
        3: "capital",
        4: "of",
        5: "France",
        6: "?",
        7: "Paris",
        8: "<EOS>",
    }

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

    final_tokens, trace = generate_toy(
        token_ids,
        transition_logits,
        eos_token_id,
        max_new_tokens=5,
    )

    print("\nPrompt:")
    print("  IDs: ", token_ids)
    print("  Text:", render_tokens(token_ids, id_to_token))

    print("\nDecoding trace:")

    for step in trace:
        selected_id = step["selected_token_id"]
        selected_text = id_to_token[selected_id]

        print(f"\nStep {step['step']}")
        print(
            "  Before:  ",
            render_tokens(step["tokens_before"], id_to_token),
        )
        print(f"  Selected: {selected_id} ({selected_text})")
        print(
            "  After:   ",
            render_tokens(step["tokens_after"], id_to_token),
        )
        print(f"  Count:    {step['generated_count']}")
        print(f"  Stop:     {step['stop']}")

    print("\nFinal sequence:")
    print("  IDs: ", final_tokens)
    print("  Text:", render_tokens(final_tokens, id_to_token))


if __name__ == "__main__":
    main()
