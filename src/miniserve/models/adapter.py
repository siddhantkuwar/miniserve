"""Assignment 6 scaffold: narrow adapter around tokenizer and pretrained model.

`mlx-lm` may load the architecture and weights. MiniServe owns prompt handling,
forward execution, generation, cache ownership, and measurements.
"""


class ModelAdapter:
    """Expose only the model capabilities MiniServe needs."""

    # TODO: Store loaded objects and reproducibility metadata without loading again.
    def __init__(self, model, tokenizer, model_id, model_revision):
        """Store loaded objects and reproducibility metadata without loading again."""
        pass

    # TODO: Load model/tokenizer once and return a configured adapter.
    @classmethod
    def load(cls, model_id, model_revision=None):
        """Load model/tokenizer once and return a configured adapter."""
        pass

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
        pass
