"""Assignment 10 test prompts for benchmark-harness correctness."""


# TODO: Use injected fake clock/evaluator to prove ordering without timing hardware.
def test_measure_once_calls_evaluate_before_stopping_clock():
    """Use injected fake clock/evaluator to prove ordering without timing hardware."""
    pass


# TODO: Plots and summaries must remain reproducible from raw data.
def test_repeated_measurement_preserves_every_raw_sample():
    """Plots and summaries must remain reproducible from raw data."""
    pass


# TODO: Prevent accidental apples-to-oranges aggregation.
def test_cold_and_warm_records_cannot_share_the_same_case_label():
    """Prevent accidental apples-to-oranges aggregation."""
    pass


# TODO: Fail when model, versions, shape, power, or swap fields are absent.
def test_benchmark_record_requires_reproducibility_metadata():
    """Fail when model, versions, shape, power, or swap fields are absent."""
    pass
