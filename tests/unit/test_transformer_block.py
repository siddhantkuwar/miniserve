"""Small, readable tests for one transformer block."""

import torch

from miniserve.engine.attention import causal_self_attention
from miniserve.engine.transformer_block import (
    feed_forward_reference,
    layer_norm_reference,
    pre_norm_transformer_block,
)


def test_layer_norm_makes_zero_mean_and_unit_variance():
    """With scale=1 and bias=0, each token should be normalized."""
    hidden_states = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]])
    scale = torch.ones(3)
    bias = torch.zeros(3)

    output = layer_norm_reference(hidden_states, scale, bias, epsilon=0.0)

    means = output.mean(dim=-1)
    variances = (output**2).mean(dim=-1)
    torch.testing.assert_close(means, torch.zeros_like(means), atol=1e-6, rtol=0)
    torch.testing.assert_close(variances, torch.ones_like(variances))


def test_feed_forward_preserves_batch_sequence_and_hidden_size():
    """The MLP may expand internally, but its output shape should match its input."""
    hidden_states = torch.randn(2, 3, 4)
    up_weight = torch.randn(4, 8)
    down_weight = torch.randn(8, 4)

    output = feed_forward_reference(hidden_states, up_weight, down_weight)

    assert output.shape == hidden_states.shape


def test_zero_sublayers_make_the_block_an_identity():
    """If attention and the MLP add zero, the residual stream should be unchanged."""
    hidden_states = torch.randn(1, 3, 4)

    def zero_attention(states, *_parameters):
        return torch.zeros_like(states)

    output = pre_norm_transformer_block(
        hidden_states,
        attention_fn=zero_attention,
        attention_parameters={
            "q_weights": torch.eye(4),
            "k_weights": torch.eye(4),
            "v_weights": torch.eye(4),
            "num_heads": 1,
        },
        mlp_parameters={
            "up_weight": torch.zeros(4, 8),
            "down_weight": torch.zeros(8, 4),
        },
        norm_parameters={
            "norm1_scale": torch.ones(4),
            "norm1_bias": torch.zeros(4),
            "norm2_scale": torch.ones(4),
            "norm2_bias": torch.zeros(4),
            "epsilon": 1e-5,
        },
    )

    torch.testing.assert_close(output, hidden_states)


def test_future_token_cannot_change_past_block_outputs():
    """The transformer block should preserve attention's causal boundary."""
    torch.manual_seed(0)
    hidden_states = torch.randn(1, 3, 4)
    changed_hidden_states = hidden_states.clone()
    changed_hidden_states[:, -1, :] = 100.0

    attention_parameters = {
        "q_weights": torch.randn(4, 4),
        "k_weights": torch.randn(4, 4),
        "v_weights": torch.randn(4, 4),
        "num_heads": 2,
    }
    mlp_parameters = {
        "up_weight": torch.randn(4, 8),
        "down_weight": torch.randn(8, 4),
    }
    norm_parameters = {
        "norm1_scale": torch.ones(4),
        "norm1_bias": torch.zeros(4),
        "norm2_scale": torch.ones(4),
        "norm2_bias": torch.zeros(4),
        "epsilon": 1e-5,
    }

    original_output = pre_norm_transformer_block(
        hidden_states,
        causal_self_attention,
        attention_parameters,
        mlp_parameters,
        norm_parameters,
    )
    changed_output = pre_norm_transformer_block(
        changed_hidden_states,
        causal_self_attention,
        attention_parameters,
        mlp_parameters,
        norm_parameters,
    )

    torch.testing.assert_close(original_output[:, :-1], changed_output[:, :-1])
