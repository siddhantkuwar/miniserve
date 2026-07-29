from sympy import true
from torch._dynamo.eval_frame import add_skip_reason
from transformers import AutoTokenizer

def inspect_special_tokens(tokenizer):
    
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
    

def inspect_round_trip(tokenizer, text):
    token_ids = tokenizer.encode(text)
    print("Token IDS: ", token_ids)
    
    decoded_text = tokenizer.decode(token_ids)
    print("Decoded text type: ", type(decoded_text))
    print("Decoded text: ", decoded_text)
    
    is_exact_match = False if (text != decoded_text) else True
    print("Exact match? ", is_exact_match)
    
    return is_exact_match


def compare_plain_and_chat_prompts(tokenizer, text):
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
    
    return length_diff


def build_shifted_next_token_pairs(tokenizer, text):
    token_ids = tokenizer.encode(text, add_special_tokens=False,)
    
    inputs = token_ids[:-1]
    labels = token_ids[1:]
    
    print("inputs: ", inputs)
    print("labels: ", labels)
    
    return inputs, labels


def main():
    text = "What is the capital of France?"
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    inspect_special_tokens(tokenizer)
    inspect_round_trip(tokenizer, text)
    compare_plain_and_chat_prompts(tokenizer, text)
    build_shifted_next_token_pairs(tokenizer, text)

if __name__ == "__main__":
    main()
