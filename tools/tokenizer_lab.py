"""Assignment 3 scaffold: inspect tokenizer and prompt-format contracts."""


# TODO: Report BOS, EOS, padding, unknown-token IDs and model max length.
def inspect_special_tokens(tokenizer):
    """Report BOS, EOS, padding, unknown-token IDs and model max length."""
    pass


# TODO: Encode and decode text, then record where round-trip text can differ.
def inspect_round_trip(tokenizer, text):
    """Encode and decode text, then record where round-trip text can differ."""
    pass


# TODO: Compare plain encoding with the model's chat-template token sequence.
def compare_plain_and_chat_prompts(tokenizer, text):
    """Compare plain encoding with the model's chat-template token sequence."""
    pass


# TODO: Return conceptual inputs `tokens[:-1]` and labels `tokens[1:]`.
def build_shifted_next_token_pairs(token_ids):
    """Return conceptual inputs `tokens[:-1]` and labels `tokens[1:]`."""
    pass


# TODO: Load only the tokenizer and print one small, annotated inspection.
def main():
    """Load only the tokenizer and print one small, annotated inspection."""
    pass
