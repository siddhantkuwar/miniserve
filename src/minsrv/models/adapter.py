from mlx_lm import load
import inspect
import mlx.core as mx

class ModelAdapter:
    def __init__(self, model, tokenizer, model_id, model_revision, config):
        """Store loaded objects and reproducibility metadata without loading again."""
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.model_revision = model_revision
        self.config = config

    @classmethod
    def load(cls, model_id, model_revision=None):
        model, tokenizer, config = load(model_id, revision=model_revision, return_config=True)
        return cls(model=model, tokenizer=tokenizer, model_id=model_id, model_revision=model_revision, config=config)

    def encode_plain_prompt(self, prompt):
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        
        if not prompt:
            raise ValueError("prompt can't be empty")
        
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        token_array = mx.array([token_ids], dtype=mx.int32)
        
        return token_array

    def encode_chat_prompt(self, messages):
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        
        if not messages:
            raise ValueError("messages can't be empty")
        
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("message must be a dictionary")
            
            if "role" not in message or "content" not in message:
                raise ValueError("message must contain a role and content")
        
        token_ids = self.tokenizer.apply_chat_template(messages, add_special_tokens=False, tokenize=True, add_generation_prompt=True)
        token_array = mx.array([token_ids], dtype=mx.int32)
                
        return token_array

    def forward(self, token_ids, cache=None):
        if token_ids.ndim != 2:
            raise ValueError(f"token_ids must be shape [B, T] -- got {token_ids.shape}")
        
        logits = self.model(token_ids, cache=cache)
        
        return {
            "logits": logits,
            "cache": cache,
        }
        
    def decode_tokens(self, token_ids):
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        
        if (isinstance(token_ids, list) and len(token_ids) == 1 and isinstance(token_ids[0], list)):
            token_ids = token_ids[0]
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def metadata(self):
        config = self.config
        
        num_attention_heads = config.get("num_attention_heads")
        hidden_size = config.get("hidden_size")

        head_dim = None
        if hidden_size is not None and num_attention_heads:
            head_dim = hidden_size // num_attention_heads

        quantization = config.get("quantization") or {}
        
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "vocab_size": config.get("vocab_size"),
            "hidden_size": hidden_size,
            "intermediate_size": config.get("intermediate_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": config.get("num_key_value_heads"),
            "head_dim": head_dim,
            "max_position_embeddings": config.get(
                "max_position_embeddings"
            ),
            "rope_theta": config.get("rope_theta"),
            "rms_norm_eps": config.get("rms_norm_eps"),
            "tie_word_embeddings": config.get(
                "tie_word_embeddings"
            ),
            "quantization_bits": quantization.get("bits"),
            "quantization_group_size": quantization.get(
                "group_size"
            ),
            "query_heads_per_kv_head": num_attention_heads // config.get("num_key_value_heads"),
            "kv_cache_elements_per_token_per_layer": 2 * config.get("num_key_value_heads") * head_dim,
        }


def main():
    #init and load
    adapter = ModelAdapter.load(
        model_id="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        model_revision="a5339a4", #this is the commit hash for the model
    )

    print("adapter type:", type(adapter))
    print("model type:", type(adapter.model))
    print("tokenizer type:", type(adapter.tokenizer))
    print("model ID:", adapter.model_id)
    print("revision:", adapter.model_revision)
    print("config type:", type(adapter.config))
    print("architecture:", adapter.config.get("architectures"))
    
    #metadata
    print("\nMetadata:")
    metadata = adapter.metadata()
    for key, value in metadata.items():
        print(f"{key}: {value}")
        
    #plain prompt
    plain_prompt = "What is the capital of France?"
    plain_ids = adapter.encode_plain_prompt(plain_prompt)

    print("\nPlain prompt:")
    print("text:", plain_prompt)
    print("token IDs:", plain_ids)
    print("shape:", plain_ids.shape)
    print("dtype:", plain_ids.dtype)

    #chat prompt
    messages = [
        {
            "role": "user",
            "content": "What is the capital of France?",
        }
    ]

    chat_ids = adapter.encode_chat_prompt(messages)

    print("\nChat prompt:")
    print("messages:", messages)
    print("token IDs:", chat_ids)
    print("shape:", chat_ids.shape)
    print("dtype:", chat_ids.dtype)
    
    #decode
    print("\nDecoded plain prompt:")
    print(adapter.decode_tokens(plain_ids))

    print("\nDecoded chat prompt:")
    print(adapter.decode_tokens(chat_ids))
    
    #forward
    forward_output = adapter.forward(chat_ids, cache=None)
    logits = forward_output["logits"]

    mx.eval(logits)

    print("\nForward output:")
    print("logits shape:", logits.shape)
    print("logits dtype:", logits.dtype)
    print("cache:", forward_output["cache"])
    
if __name__ == "__main__":
    main()