"""Assignment 3 scaffold: inspect tokenizer and prompt-format contracts."""
from sympy import true
from torch._dynamo.eval_frame import add_skip_reason
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
    plain_ids = tokenizer.encode(text, add_special_tokens=False)
    print("plain ids: ", plain_ids)
    print("plain token count: ", len(plain_ids))
    
    msgs = [
        {
            "role": "user",
            "content": text
        }
    ]
    print("msgs: ", msgs)
    
    rendered_chat = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print("rendered chat: ", repr(rendered_chat))
    
    chat_ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    print("chat ids: ", chat_ids)
    print("chat token count: ", len(chat_ids))
    
    print("chat_ids type: ", type(chat_ids))
    print("chat_ids keys: ", chat_ids.keys())
    
    chat_token_ids = chat_ids["input_ids"]
    print("The actual length of chat_ids: ", len(chat_token_ids))
    
    same_ids = True if (plain_ids == chat_token_ids) else False
    print("Same ID? ", same_ids)
    
    length_diff = len(chat_token_ids) - len(plain_ids)
    print("length difference: ", length_diff)
    
    return plain_ids


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
    compare_plain_and_chat_prompts(tokenizer, text)

if __name__ == "__main__":
    main()
