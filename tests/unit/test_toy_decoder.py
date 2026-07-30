"""Small, readable tests for the toy autoregressive decoder."""

import torch

from miniserve.engine.toy_decoder import generate_toy, select_greedy_token


def _transition_table():
    """Create the predictable path 0 -> 1 -> 2 -> 3, where 3 is EOS."""
    table = torch.full((4, 4), -10.0)
    table[0, 1] = 10.0
    table[1, 2] = 10.0
    table[2, 3] = 10.0
    table[3, 3] = 10.0
    return table


def test_greedy_selection_returns_highest_logit_index():
    """Greedy decoding should select the position with the largest score."""
    logits = torch.tensor([-2.0, 5.0, 1.0])

    selected_token = select_greedy_token(logits)

    assert selected_token == 1


def test_generation_appends_exactly_one_token_per_step():
    """Each trace entry should show one new token added to the sequence."""
    transition_logits = _transition_table()
    trace = generate_toy(
        token_ids=[0],
        transition_logits=transition_logits,
        eos_token_id=3,
        max_new_tokens=2,
    )

    assert len(trace) == 2
    assert trace[0]["sequence_before"] == [0]
    assert trace[0]["selected_token"] == 1
    torch.testing.assert_close(trace[0]["logits"], transition_logits[0])
    assert trace[1]["sequence_before"] == [0, 1]
    assert trace[1]["selected_token"] == 2


def test_generation_stops_on_eos():
    """EOS should stop generation before the token budget is exhausted."""
    trace = generate_toy(
        token_ids=[0],
        transition_logits=_transition_table(),
        eos_token_id=3,
        max_new_tokens=10,
    )

    assert len(trace) == 3
    assert trace[-1]["selected_token"] == 3
    assert trace[-1]["stop_reason"] == "eos"


def test_generation_stops_at_max_new_tokens_without_eos():
    """The explicit token budget should prevent an infinite loop."""
    trace = generate_toy(
        token_ids=[0],
        transition_logits=_transition_table(),
        eos_token_id=99,
        max_new_tokens=2,
    )

    assert len(trace) == 2
    assert trace[-1]["stop_reason"] == "max_new_tokens"


def test_same_input_produces_the_same_greedy_trace():
    """Greedy generation should be deterministic for a fixed transition table."""
    first_trace = generate_toy([0], _transition_table(), eos_token_id=3, max_new_tokens=4)
    second_trace = generate_toy([0], _transition_table(), eos_token_id=3, max_new_tokens=4)

    first_tokens = [step["selected_token"] for step in first_trace]
    second_tokens = [step["selected_token"] for step in second_trace]

    assert first_tokens == second_tokens
