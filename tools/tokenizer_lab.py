"""Assignment 3 scaffold: inspect tokenizer and prompt-format contracts."""
from transformers import AutoTokenizer

# TODO: Report BOS, EOS, padding, unknown-token IDs and model max length.
def inspect_special_tokens(tokenizer):
    """Report BOS, EOS, padding, unknown-token IDs and model max length."""
    
    '''
    noob way of doing ts:
    
    bos = tokenizer.bos_token
    bos_id = tokenizer.bos_token_id
    
    eos = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id
    
    pad = tokenizer.pad_token
    pad_id = tokenizer.pad_token_id
    
    unk = tokenizer.unk_token
    unk_id = tokenizer.unk_token_id
    
    model_max_length = tokenizer.model_max_length
    
    print("BOS Token: ", bos)
    print("BOS Token ID: ", bos_id)
    print("EOS Token: ", eos)
    print("EOS Token ID: ", eos_id)
    print("PAD Token: ", pad)
    print("PAD Token ID: ", pad_id)
    print("UNK Token: ", unk)
    print("UNK Token ID: ", unk_id)
    print("model max length: ", model_max_length)
    '''
    
    #better way
    report = {
        "bos": {
            "token": tokenizer.bos_token,
            "id": tokenizer.bos_token_id,
        },
        "eos": {
            "token": tokenizer.eos_token,
            "id": tokenizer.eos_token_id,
        },
        "pad": {
            "token": tokenizer.pad_token,
            "id": tokenizer.pad_token_id,
        },
        "unk": {
            "token": tokenizer.unk_token,
            "id": tokenizer.unk_token_id,
        },
        "model_max_length": tokenizer.model_max_length,
    }
    
    print("BOS:", report["bos"])
    print("EOS:", report["eos"])
    print("PAD:", report["pad"])
    print("UNK:", report["unk"])
    print("Model max length:", report["model_max_length"])
    
    return report
    

# TODO: Encode and decode text, then record where round-trip text can differ.
def inspect_round_trip(tokenizer, text):
    """Encode and decode text, then record where round-trip text can differ."""
    token_ids = tokenizer.encode(text)
    print("Token IDS: ", token_ids)
    
    decoded_text = tokenizer.decode(token_ids)
    print("Decoded text type: ", type(decoded_text))
    print("Decoded text: ", decoded_text)
    
    is_exact_match = False if (text != decoded_text) else True
    print("Exact match? ", is_exact_match)
    
    return is_exact_match


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
    
    text = "What is the capital of France?"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    inspect_special_tokens(tokenizer)
    inspect_round_trip(tokenizer, text)
    

if __name__ == "__main__":
    main()
