import torch
import torch.nn as nn

"""Assignment 1: your causal multi-head self-attention reference.

This file is intentionally a scaffold. Do not look for a hidden solution here:
you will write the implementation yourself as you work through the assignment.

Suggested construction order:

1. Write a private `_softmax` helper operating over the final axis.
2. Define `causal_self_attention(...)` with inputs for hidden states, Q/K/V
   weights, and `num_heads`.
3. Validate that hidden states are `[batch, sequence, hidden]` and that the
   hidden dimension divides evenly into heads.
4. Project hidden states into queries, keys, and values.
5. Split the hidden dimension into `[heads, head_dim]`, then transpose so the
   head axis comes before the sequence axis.
6. Calculate scaled query-key dot-product scores.
7. Apply a causal mask before softmax: position i may read keys 0 through i.
8. Mix value vectors using the attention probabilities.
9. Reassemble the per-head output into `[batch, sequence, hidden]`.

Write down every tensor shape beside your code while you work. A correct,
readable reference matters more than compact code or performance here.
"""

# function to recieve hidden_states and it has the shape of [B, T, D]
'''
1. Check that hidden_states has exactly 3 dimensions.
2. Read B, T, and D from its shape.
3. Check that D divides evenly by num_heads.
4. Calculate Dh = D // num_heads.
'''

hidden_states = torch.zeros((2, 4, 8))

def validateAttentionInput(hidden_states, num_heads):
   # check that hidden_states has exactly 3 dimensions
   if hidden_states.ndim != 3:
      raise ValueError("hidden_states must have exactly 3 dimensions")
   
   # read B, T, and D from its shape
   B, T, D = hidden_states.shape
   
   # validate num_heads bc it cant when its 0 or -2 for some reason
   if num_heads <= 0:
      raise ValueError("num_heads must be a positive integer")
   
   # check that D divides evenly by num_heads
   if D % num_heads != 0:
      raise ValueError("D must divide evenly by num_heads")
   
   # calculate Dh = D // num_heads
   Dh = D // num_heads
   
   return B, T, D, Dh

   