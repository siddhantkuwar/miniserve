"""Assignment 6 integration-test prompts. Model downloads must be explicit."""


# TODO: Assert integer token dtype and `[1, sequence]` shape.
def test_plain_prompt_encoding_has_batch_and_sequence_axes():
    """Assert integer token dtype and `[1, sequence]` shape."""
    pass


# TODO: Relate vocabulary dimension to tokenizer metadata.
def test_forward_logits_have_batch_sequence_vocabulary_axes():
    """Relate vocabulary dimension to tokenizer metadata."""
    pass


# TODO: A benchmark is not reproducible without model/runtime identity.
def test_model_metadata_records_exact_identity_and_runtime_details():
    """A benchmark is not reproducible without model/runtime identity."""
    pass


# TODO: Protect the boundary: loading/oracle access must not become runtime generation.
def test_adapter_does_not_expose_generation_helper():
    """Protect the boundary: loading/oracle access must not become runtime generation."""
    pass
