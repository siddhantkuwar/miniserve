"""Assignment 6 scaffold: narrow adapter around tokenizer and pretrained model.

`mlx-lm` may load the architecture and weights. MiniServe owns prompt handling,
forward execution, generation, cache ownership, and measurements.
"""
from mlx_lm import load
import inspect

#print(inspect.getsource(load))

class ModelAdapter:
    """Expose only the model capabilities MiniServe needs."""

    # TODO: Store loaded objects and reproducibility metadata without loading again.
    def __init__(self, model, tokenizer, model_id, model_revision, config):
        """Store loaded objects and reproducibility metadata without loading again."""
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.model_revision = model_revision
        self.config = config

    # TODO: Load model/tokenizer once and return a configured adapter.
    @classmethod
    def load(cls, model_id, model_revision=None):
        """Load model/tokenizer once and return a configured adapter."""
        model, tokenizer, config = load(model_id, revision=model_revision, return_config=True)
        return cls(model=model, tokenizer=tokenizer, model_id=model_id, model_revision=model_revision, config=config)

    # TODO: Convert a plain prompt to a batch-shaped MLX token array.
    def encode_plain_prompt(self, prompt):
        """Convert a plain prompt to a batch-shaped MLX token array."""
        pass

    # TODO: Apply the tokenizer chat template explicitly and return token IDs.
    def encode_chat_prompt(self, messages):
        """Apply the tokenizer chat template explicitly and return token IDs."""
        pass

    # TODO: Run exactly one model forward call and return normalized logits/cache output.
    def forward(self, token_ids, cache=None):
        """Run exactly one model forward call and return normalized logits/cache output."""
        pass

    # TODO: Convert token IDs to text with a declared special-token policy.
    def decode_tokens(self, token_ids):
        """Convert token IDs to text with a declared special-token policy."""
        pass

    # TODO: Return model ID, revision, dtype/quantization, vocabulary, and dimensions.
    def metadata(self):
        """Return model ID, revision, dtype/quantization, vocabulary, and dimensions."""
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
        }


def main():
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
    
    print("\nMetadata:")
    metadata = adapter.metadata()
    for key, value in metadata.items():
        print(f"{key}: {value}")
    
if __name__ == "__main__":
    main()