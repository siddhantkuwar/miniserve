from sympy import approximants
import torch
import torch.nn.functional as F
import mlx.core as mx
import mlx.nn as nn

from miniserve.engine.transformer_block import down_weight


# TODO: Run the tiny MLP using PyTorch tensor operations and fixed parameters.
def build_equivalent_torch_mlp(parameters, inputs):
    """Run the tiny MLP using PyTorch tensor operations and fixed parameters."""
    up_weight = parameters["up_weight"]
    down_weight = parameters["down_weight"]
    
    expanded = inputs @ up_weight
    activated = F.GELU(expanded, approximate=None)
    output = activated @ down_weight
    
    return output


# TODO: Run the same equations and parameters with MLX arrays.
def build_equivalent_mlx_mlp(parameters, inputs):
    """Run the same equations and parameters with MLX arrays."""
    


# TODO: Separate graph construction from execution and identify forced evaluation.
def demonstrate_lazy_evaluation(array):
    """Separate graph construction from execution and identify forced evaluation."""
    pass


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
    pass
