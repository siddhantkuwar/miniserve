import torch
import torch.nn.functional as F
import mlx.core as mx
import mlx.nn as nn
import time


def build_equivalent_torch_mlp(parameters, inputs):
    up_weight = parameters["up_weight"]
    down_weight = parameters["down_weight"]
    
    expanded = inputs @ up_weight
    activated = F.gelu(expanded, approximate='none')
    output = activated @ down_weight
    
    return output


def build_equivalent_mlx_mlp(parameters, inputs):
    up_weight = parameters["up_weight"]
    down_weight = parameters["down_weight"]
    
    expanded = inputs @ up_weight
    activated = nn.gelu(expanded)
    output = activated @ down_weight
    
    return output


def demonstrate_lazy_evaluation(array):
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


def run_on_stream(operation, inputs, stream):
    with mx.stream(stream):
        result = operation(inputs)

    return result


def time_with_evaluation(operation, inputs, warmups, repeats):
    samples = []

    for _ in range(warmups):
        result = operation(inputs)
        mx.eval(result)

    for _ in range(repeats):
        start = time.perf_counter()

        result = operation(inputs)
        mx.eval(result)

        end = time.perf_counter()

        samples.append(end - start)

    return samples


def main():
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
    print("\n")
    
    
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
    print("\nResult:")
    print(lazy_demo["result"])
    
    cpu_stream = mx.default_stream(mx.cpu)
    gpu_stream = mx.default_stream(mx.gpu)

    cpu_output = run_on_stream(
        lambda x: build_equivalent_mlx_mlp(mlx_parameters, x),
        mlx_inputs,
        cpu_stream,
    )

    gpu_output = run_on_stream(
        lambda x: build_equivalent_mlx_mlp(mlx_parameters, x),
        mlx_inputs,
        gpu_stream,
    )

    mx.eval(cpu_output, gpu_output)

    print("CPU output:")
    print(cpu_output)

    print("GPU output:")
    print(gpu_output)
    
    timing_samples = time_with_evaluation(
        lambda x: build_equivalent_mlx_mlp(mlx_parameters, x),
        mlx_inputs,
        warmups=3,
        repeats=10,
    )

    print("\nTrue MLX Timing: ")

    for index, sample in enumerate(timing_samples, start=1):
        print(f"Run {index}: {sample * 1_000:.6f} ms")

    print(
        f"Average: "
        f"{sum(timing_samples) / len(timing_samples) * 1_000:.6f} ms"
    )


if __name__ == "__main__":
    main()
