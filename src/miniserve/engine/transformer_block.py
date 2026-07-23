"""Assignment 2 scaffold: transformer block anatomy in PyTorch.

Implement a small pre-normalization transformer block only after you can draw
the residual stream and state the shape at every boundary. This is a learning
oracle, not MiniServe's final MLX model implementation.
"""
import torch


# TODO: Normalize the final hidden dimension and apply learned scale/bias.
def layer_norm_reference(hidden_states, scale, bias, epsilon):
    """Normalize the final hidden dimension and apply learned scale/bias.

    Questions to answer first:
    - Which dimensions are reduced to compute mean and variance?
    - Why does epsilon exist?
    - Which shape must scale and bias have for broadcasting?
    """
    
    avg = torch.mean(hidden_states, dim=-1, keepdim=True)  # we want to first avg out the D vector, while 
    
    center = hidden_states - avg
    var = torch.mean(center ** 2, dim=-1, keepdim=True)
    denom = torch.sqrt(var + epsilon)
    
    normal = center / denom
    output = normal * scale + bias
    
    '''
    print("hidden_states shape:", hidden_states.shape)
    print("avg shape:", avg.shape)
    print("center shape:", center.shape)
    print("checking if value is 0 and shape is [2, 3]: ", center.mean(dim=-1))
    print("var shape:", var.shape)
    print("normalized mean:", normal.mean(dim=-1))
    print("normalized variance:", torch.mean(normal ** 2, dim=-1))
    '''
    
    return output


# TODO: Apply hidden expansion, GELU, and projection back to model width.
def feed_forward_reference(hidden_states, up_weight, down_weight):
    """Apply hidden expansion, GELU, and projection back to model width.

    Track `[batch, sequence, hidden] -> [..., mlp_hidden] -> [..., hidden]`.
    """
    pass


# TODO: Compose pre-norm attention, residuals, pre-norm MLP, and residuals.
def pre_norm_transformer_block(
    hidden_states,
    attention_fn,
    attention_parameters,
    mlp_parameters,
    norm_parameters,
):
    """Compose pre-norm attention, residuals, pre-norm MLP, and residuals.

    Keep each residual addition visible. Do not hide the order inside a generic
    sequential container until you can explain why pre-norm differs from
    post-norm.
    """
    pass

if __name__ == "__main__":
    B, T, D = 2, 3, 4
    
    hidden_states = torch.randn(B, T, D)
    
    scale = torch.ones(D)
    bias = torch.zeros(D)
    epsilon = 1e-5
    
    output = layer_norm_reference(hidden_states, scale, bias, epsilon)
    
    print("output shape: ", output.shape)