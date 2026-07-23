"""Assignment 10 scaffold: truthful timing primitives for lazy execution."""


# TODO: Run the exact benchmark shape repeatedly and force each result as designed.
def warmup(operation, inputs, iterations, evaluate):
    """Run the exact benchmark shape repeatedly and force each result as designed."""
    pass


# TODO: Time completed execution—not only lazy graph construction—and return nanoseconds.
def measure_once(operation, inputs, evaluate, clock):
    """Time completed execution—not only lazy graph construction—and return nanoseconds."""
    pass


# TODO: Return every raw sample plus enough shape metadata to group valid comparisons.
def measure_repeated(operation, inputs, warmup_count, repeat_count, evaluate, clock):
    """Return every raw sample plus enough shape metadata to group valid comparisons."""
    pass


# TODO: Calculate median and percentiles while preserving raw values separately.
def summarize_samples(samples):
    """Calculate median and percentiles while preserving raw values separately."""
    pass


# TODO: Append one reproducible benchmark record with hardware/runtime metadata.
def write_jsonl_record(path, record):
    """Append one reproducible benchmark record with hardware/runtime metadata."""
    pass
