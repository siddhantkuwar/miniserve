import torch
import math

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

def _validate_attention_input(hidden_states, num_heads):
   # check that hidden_states has exactly 3 dimensions
   if hidden_states.ndim != 3:
      raise ValueError("hidden_states must have exactly 3 dimensions")
   
   # read B, T, and D from its shape
   B, T, D = hidden_states.shape
   
   # validate num_heads since head count must be positive to recieve hidden features
   if num_heads <= 0:
      raise ValueError("num_heads must be a positive integer")
   
   # check that D divides evenly by num_heads
   if D % num_heads != 0:
      raise ValueError("D must divide evenly by num_heads")
   
   # calculate Dh = D // num_heads
   Dh = D // num_heads
   
   return B, T, D, Dh


# function to convert scores into normalized probabilities along the final axis. 
'''
1. subtract maximum for numerical stability
2. exponentiate
3. sum over the final dimension
4. divide by the sum
5. return
'''
def _softmax_last_dim(x):
   
   # calculate the maximum value along the last dimension
   max_value = torch.amax(x, dim=-1, keepdim=True)
   
   # subtract the maximum value from x
   x = x - max_value
   
   # apply the exponential function to x
   x = torch.exp(x)
   
   # sum the exponential values along the last dimension
   sum_exp = x.sum(dim=-1, keepdim=True)
   
   # divide x by the sum of the exponential values
   x = x / sum_exp
   
   return x


#function that contains the entire causal self-attention computation
'''
1. validate input
2. project Q/K/V, shape of [B, T, D] -> [B, T, num_heads, Dh]
3. split heads, shape of [B, T, num_heads, Dh] -> [B, num_heads, T, Dh]
4. compute scaled scores, shape of [B, num_heads, T, Dh]
5. apply causal mask, shape of [B, num_heads, T, Dh]
6. softmax, shape of [B, num_heads, T, Dh]
7. mix values, shape of [B, num_heads, T, Dh]
8. merge heads, shape of [B, T, D]
9. return
'''

def causal_self_attention(hidden_states, q_weights, k_weights, v_weights, num_heads):
   
   # validate input
   B, T, D, Dh = _validate_attention_input(hidden_states, num_heads)
   
   # project Q/K/V
   Q = torch.matmul(hidden_states, q_weights)
   K = torch.matmul(hidden_states, k_weights)
   V = torch.matmul(hidden_states, v_weights)
   
   # split heads
   Q = Q.view(B, T, num_heads, Dh)
   K = K.view(B, T, num_heads, Dh)
   V = V.view(B, T, num_heads, Dh)
   
   Q = Q.transpose(1, 2)  # [B, T, H, Dh] -> [B, H, T, Dh]
   K = K.transpose(1, 2)
   V = V.transpose(1, 2)
   
   print("Q:", Q.shape)
   print("K:", K.shape)
   print("V:", V.shape)
   
   # compute scaled scores
   scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Dh)
   
   # apply causal mask
   mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=scores.device))
   scores = scores.masked_fill(~mask, float("-inf"))
   
   # softmax
   scores = _softmax_last_dim(scores)
   
   # mix values
   output = torch.matmul(scores, V)
   
   #transpose and reshape
   output = output.transpose(1, 2)  # [B, H, T, Dh] -> [B, T, H, Dh]
   output = output.reshape(B, T, D)
   
   return output

#main
if __name__ == "__main__":
   B, T, D = 2, 4, 8
   num_heads = 2

   hidden_states = torch.randn(B, T, D)

   q_weights = torch.randn(D, D)
   k_weights = torch.randn(D, D)
   v_weights = torch.randn(D, D)

   output = causal_self_attention(hidden_states, q_weights, k_weights, v_weights, num_heads,)

   print("output:", output.shape)