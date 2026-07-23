"""Assignment 10 scaffold: uncached decoding benchmark entry point."""


# TODO: Measure adapter/model initialization separately from inference.
def benchmark_model_load(config):
    """Measure adapter/model initialization separately from inference."""
    pass


# TODO: Capture first graph/kernel/shape costs without mixing warm measurements.
def benchmark_cold_first_token(adapter, prompt_tokens, config):
    """Capture first graph/kernel/shape costs without mixing warm measurements."""
    pass


# TODO: Measure full prompt processing after exact-shape warmup.
def benchmark_warm_prefill(adapter, prompt_tokens, config):
    """Measure full prompt processing after exact-shape warmup."""
    pass


# TODO: Measure inter-token latency and tokens/sec for the uncached loop.
def benchmark_warm_decode(adapter, prompt_tokens, config):
    """Measure inter-token latency and tokens/sec for the uncached loop."""
    pass


# TODO: Record chip, power, swap, versions, model identity, shapes, and quantization.
def collect_environment_metadata():
    """Record chip, power, swap, versions, model identity, shapes, and quantization."""
    pass


# TODO: Run selected cases and write raw local JSONL/CSV before any plots.
def main():
    """Run selected cases and write raw local JSONL/CSV before any plots."""
    pass
