"""Assignment 16 scaffold: compare sequential and concurrent/static-batch service use."""


# TODO: Consume one stream and return token events plus client-observed timings.
async def submit_one(client, request_payload):
    """Consume one stream and return token events plus client-observed timings."""
    pass


# TODO: Submit one request at a time as the latency/throughput baseline.
async def run_sequential(client, payloads):
    """Submit one request at a time as the latency/throughput baseline."""
    pass


# TODO: Submit N requests concurrently while preserving per-request results.
async def run_concurrent(client, payloads):
    """Submit N requests concurrently while preserving per-request results."""
    pass


# TODO: Report aggregate throughput alongside every request's TTFT/latency.
def summarize_load_run(results):
    """Report aggregate throughput alongside every request's TTFT/latency."""
    pass


# TODO: Run declared workload cases and write raw local result records.
def main():
    """Run declared workload cases and write raw local result records."""
    pass
