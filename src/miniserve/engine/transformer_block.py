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
    expanded = hidden_states @ up_weight
    #print("hidden_states shape:", hidden_states.shape)
    #print("up_weight shape:", up_weight.shape)
    #print("expanded shape:", expanded.shape)
    
    '''
    test_gelu = torch.randn(2, 3, 12)
    test_gelu_output = torch.nn.functional.gelu(test_gelu)
    
    print("test input shape: ", test_gelu.shape)
    print("test gelu output shape: ", test_gelu_output.shape)
    
    test_gelu2 = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    gelu_output2 = torch.nn.functional.gelu(test_gelu2)

    print("input:", test_gelu2)
    print("output:", gelu_output2)
    '''
    
    activated = torch.nn.functional.gelu(expanded)
    #print("activated shape: ", activated.shape)
    
    mlp_output = activated @ down_weight
    #print("shrinked shape: ", shrinked.shape)
    
    return mlp_output


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
    
    M = 12
    up_weight = torch.randn(D, M)
    down_weight = torch.randn(M, D)
    
    output = layer_norm_reference(hidden_states, scale, bias, epsilon)
    mlp_output = feed_forward_reference(hidden_states, up_weight, down_weight)
    
    print("output shape: ", output.shape)
    print("mlp_output shape: ", mlp_output.shape)