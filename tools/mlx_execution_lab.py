"""Assignment 5 scaffold: compare PyTorch eager execution with MLX execution."""


# TODO: Run the tiny MLP using PyTorch tensor operations and fixed parameters.
def build_equivalent_torch_mlp(parameters, inputs):
    """Run the tiny MLP using PyTorch tensor operations and fixed parameters."""
    pass


# TODO: Run the same equations and parameters with MLX arrays.
def build_equivalent_mlx_mlp(parameters, inputs):
    """Run the same equations and parameters with MLX arrays."""
    pass


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
