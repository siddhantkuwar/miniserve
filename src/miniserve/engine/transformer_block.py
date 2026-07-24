#description + comments needed

import torch
from attention import causal_self_attention

def layer_norm_reference(hidden_states, scale, bias, epsilon):
    
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


def feed_forward_reference(hidden_states, up_weight, down_weight):

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


def pre_norm_transformer_block(
    hidden_states,
    attention_fn,
    attention_parameters,
    mlp_parameters,
    norm_parameters,
):
    
    #unpacking all the params
    q_weights = attention_parameters["q_weights"]
    k_weights = attention_parameters["k_weights"]
    v_weights = attention_parameters["v_weights"]
    num_heads = attention_parameters["num_heads"]
    
    up_weight = mlp_parameters["up_weight"]
    down_weight = mlp_parameters["down_weight"]

    norm1_scale = norm_parameters["norm1_scale"]
    norm1_bias = norm_parameters["norm1_bias"]

    norm2_scale = norm_parameters["norm2_scale"]
    norm2_bias = norm_parameters["norm2_bias"]

    epsilon = norm_parameters["epsilon"]
    
    #LN1
    normalized_x = layer_norm_reference(hidden_states, norm1_scale, norm1_bias, epsilon)
    #print("normalized_x shape: ", normalized_x.shape)
    
    attention_update = attention_fn(normalized_x, q_weights, k_weights, v_weights, num_heads)
    #print("attention update shape: ", attention_update.shape)
    
    x1 = hidden_states + attention_update
    #print("x1 shape: ", x1.shape)
    
    #LN2
    second_norm_x = layer_norm_reference(x1, norm2_scale, norm2_bias, epsilon)
    #print("second_norm_x shape: ", second_norm_x.shape)
    
    mlp_update = feed_forward_reference(second_norm_x, up_weight, down_weight)
    #print("mlp_update shape: ", mlp_update.shape)
    
    x2 = x1 + mlp_update
    #print("x2 shape: ", x2.shape)
    
    return x2


#main
if __name__ == "__main__":
    B, T, D = 2, 3, 4
    
    hidden_states = torch.randn(B, T, D)
    
    scale = torch.ones(D)
    bias = torch.zeros(D)
    epsilon = 1e-5
    
    M = 12
    up_weight = torch.randn(D, M)
    down_weight = torch.randn(M, D)
    
    q_weight = torch.randn(D, D)
    k_weight = torch.randn(D, D)
    v_weight = torch.randn(D, D)
    num_heads = 1
    
    output = layer_norm_reference(hidden_states, scale, bias, epsilon)
    mlp_output = feed_forward_reference(hidden_states, up_weight, down_weight)
    norm_output = pre_norm_transformer_block(
        hidden_states,
        attention_fn=causal_self_attention,
        attention_parameters={
            "q_weights": q_weight,
            "k_weights": k_weight,
            "v_weights": v_weight,
            "num_heads": num_heads,
        },
        mlp_parameters={
            "up_weight": up_weight,
            "down_weight": down_weight,
        },
        norm_parameters={
            "norm1_scale": scale,
            "norm1_bias": bias,
            "norm2_scale": scale,
            "norm2_bias": bias,
            "epsilon": epsilon,
        },
    )
    
    print("output shape: ", output.shape)
    print("mlp_output shape: ", mlp_output.shape)
    print("norm_output shape: ", norm_output.shape)