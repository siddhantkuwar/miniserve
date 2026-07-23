"""Assignment 13 scaffold: empirical host/MLX memory-bandwidth measurements."""


# TODO: Select cache-exceeding arrays conservatively without causing swap.
def choose_working_set_sizes(available_memory_bytes):
    """Select cache-exceeding arrays conservatively without causing swap."""
    pass


# TODO: Measure completed copies, return raw samples, and label CPU/GPU stream.
def benchmark_copy(size_bytes, stream, warmups, repeats):
    """Measure completed copies, return raw samples, and label CPU/GPU stream."""
    pass


# TODO: Measure a read-heavy reduction separately from copy bandwidth.
def benchmark_read_reduction(size_bytes, stream, warmups, repeats):
    """Measure a read-heavy reduction separately from copy bandwidth."""
    pass


# TODO: Record macOS pressure/swap before and after each benchmark case.
def sample_memory_pressure_and_swap():
    """Record macOS pressure/swap before and after each benchmark case."""
    pass


# TODO: Run safe sizes, reject swapped results, and write raw local evidence.
def main():
    """Run safe sizes, reject swapped results, and write raw local evidence."""
    pass
