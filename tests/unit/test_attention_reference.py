"""Small, readable tests for causal self-attention."""

import pytest
import torch

from miniserve.engine.attention import (
    _softmax_last_dim,
    _validate_attention_input,
    causal_self_attention,
)


def test_attention_output_has_the_same_shape_as_the_input():
    """Attention should return one output vector for every input token."""
    hidden_states = torch.randn(2, 3, 4)
    identity = torch.eye(4)

    output = causal_self_attention(
        hidden_states,
        identity,
        identity,
        identity,
        num_heads=2,
    )

    assert output.shape == hidden_states.shape


def test_softmax_rows_sum_to_one():
    """Each row produced by softmax should be a probability distribution."""
    scores = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]])

    probabilities = _softmax_last_dim(scores)
    row_sums = probabilities.sum(dim=-1)

    torch.testing.assert_close(row_sums, torch.ones(2))


def test_future_token_cannot_change_past_outputs():
    """Changing the final token must not change outputs for earlier positions."""
    hidden_states = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]
    )
    changed_hidden_states = hidden_states.clone()
    changed_hidden_states[:, -1, :] = torch.tensor([100.0, -100.0])
    identity = torch.eye(2)

    original_output = causal_self_attention(
        hidden_states,
        identity,
        identity,
        identity,
        num_heads=1,
    )
    changed_output = causal_self_attention(
        changed_hidden_states,
        identity,
        identity,
        identity,
        num_heads=1,
    )

    torch.testing.assert_close(original_output[:, :-1], changed_output[:, :-1])


def test_invalid_number_of_heads_raises_a_clear_error():
    """The hidden size must divide evenly across all attention heads."""
    hidden_states = torch.zeros(1, 3, 4)

    with pytest.raises(ValueError, match="divide evenly"):
        _validate_attention_input(hidden_states, num_heads=3)
