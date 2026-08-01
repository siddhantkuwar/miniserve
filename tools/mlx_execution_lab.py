import torch
import torch.nn.functional as F
import mlx.core as mx
import mlx.nn as nn
import time


# TODO: Run the tiny MLP using PyTorch tensor operations and fixed parameters.
def build_equivalent_torch_mlp(parameters, inputs):
    """Run the tiny MLP using PyTorch tensor operations and fixed parameters."""
    up_weight = parameters["up_weight"]
    down_weight = parameters["down_weight"]
    
    expanded = inputs @ up_weight
    activated = F.gelu(expanded, approximate='none')
    output = activated @ down_weight
    
    return output


# TODO: Run the same equations and parameters with MLX arrays.
def build_equivalent_mlx_mlp(parameters, inputs):
    """Run the same equations and parameters with MLX arrays."""
    up_weight = parameters["up_weight"]
    down_weight = parameters["down_weight"]
    
    expanded = inputs @ up_weight
    activated = nn.gelu(expanded)
    output = activated @ down_weight
    
    return output


# TODO: Separate graph construction from execution and identify forced evaluation.
def demonstrate_lazy_evaluation(array):
    """Separate graph construction from execution and identify forced evaluation."""
    construction_start = time.perf_counter()
    
    result = mx.square(array) + 1.0
    
    construction_end = time.perf_counter()
    eval_start = time.perf_counter()
    
    mx.eval(result)
    
    eval_end = time.perf_counter()
    
    return {
        "result": result,
        "construction_seconds": construction_end - construction_start,
        "evaluation_seconds": eval_end - eval_start,
    }


# TODO: Execute the same operation on MLX CPU and GPU streams without copying arrays.
def run_on_stream(operation, inputs, stream):
    """Execute the same operation on MLX CPU and GPU streams without copying arrays."""
    pass


# TODO: Warm the exact shape, force `mx.eval`, and return raw elapsed samples.
def time_with_evaluation(operation, inputs, warmups, repeats):
    """Warm the exact shape, force `mx.eval`, and return raw elapsed samples."""
    pass


# TODO: Print parity, device, lazy-evaluation, and bad-vs-correct timing evidence.
def main():
    """Print parity, device, lazy-evaluation, and bad-vs-correct timing evidence."""
    torch_inputs = torch.tensor(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ],
        dtype=torch.float32,
    )
    
    torch_parameters = {
        "up_weight": torch.tensor(
            [
                [0.5, -0.5, 1.0],
                [1.0, 0.5, -1.0],
            ],
            dtype=torch.float32,
        ),
        
        "down_weight": torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, -0.5],
            ],
            dtype=torch.float32,
        ),
    }
    
    torch_output = build_equivalent_torch_mlp(
        torch_parameters,
        torch_inputs
    )
    
    print("Input shape:", torch_inputs.shape)
    print("Output shape:", torch_output.shape)
    print("Torch output:")
    print(torch_output)
    print("\n")
    
    mlx_inputs = mx.array(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ],
        dtype=mx.float32,
    )

    mlx_parameters = {
        "up_weight": mx.array(
            [
                [0.5, -0.5, 1.0],
                [1.0, 0.5, -1.0],
            ],
            dtype=mx.float32,
        ),
        
        "down_weight": mx.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, -0.5],
            ],
            dtype=mx.float32,
        ),
    }
    
    mlx_output = build_equivalent_mlx_mlp(
        mlx_parameters,
        mlx_inputs,
    )
    
    mx.eval(mlx_output)
    
    print("MLX input shape:", mlx_inputs.shape)
    print("MLX output shape:", mlx_output.shape)
    print("MLX output:")
    print(mlx_output)
    
    
    lazy_demo = demonstrate_lazy_evaluation(mlx_inputs)

    print(
        "Graph construction:",
        lazy_demo["construction_seconds"],
        "seconds",
    )
    print(
        "Forced evaluation:",
        lazy_demo["evaluation_seconds"],
        "seconds",
    )
    print("Result:")
    print(lazy_demo["result"])


if __name__ == "__main__":
    main()
