"""
Attention mechanisms for sequence-to-sequence modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        Q: Query tensor [batch, ..., seq_len_q, d_k]
        K: Key tensor [batch, ..., seq_len_k, d_k]
        V: Value tensor [batch, ..., seq_len_v, d_k]
        mask: Optional mask [batch, ..., seq_len_q, seq_len_k]
              Values: 1 for positions to attend, 0 for positions to mask

    Returns:
        output: Attention output [batch, ..., seq_len_q, d_k]
        attention_weights: Attention weights [batch, ..., seq_len_q, seq_len_k]
    """
    d_k = Q.size(-1)

    # TODO: Compute attention scores
    # TODO: Scale scores
    # TODO: Apply mask if provided (use masked_fill to set masked positions to -inf)
    # TODO: Apply softmax
    # TODO: Apply attention to values

    # TODO: Compute attention scores
    # [batch, ..., seq_len_q, d_k] @ [batch, ..., d_k, seq_len_k]
    # -> [batch, ..., seq_len_q, seq_len_k]
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # TODO: Scale scores
    scores = scores / math.sqrt(d_k)

    # TODO: Apply mask if provided (use masked_fill to set masked positions to -inf)
    if mask is not None:
        # We assume mask has 0s where we want to mask and 1s otherwise
        # masked_fill fills elements where the condition (mask == 0) is True
        scores = scores.masked_fill(mask == 0, -1e9)

    # TODO: Apply softmax
    # Softmax over the last dimension (seq_len_k)
    attention_weights = F.softmax(scores, dim=-1)

    # TODO: Apply attention to values
    # [batch, ..., seq_len_q, seq_len_k] @ [batch, ..., seq_len_v, d_k]
    # Note: seq_len_k == seq_len_v
    # -> [batch, ..., seq_len_q, d_k]
    output = torch.matmul(attention_weights, V)

    return output, attention_weights

    # raise NotImplementedError


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    Splits d_model into num_heads, applies attention in parallel,
    then concatenates and projects the results.
    """

    def __init__(self, d_model, num_heads):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # TODO: Initialize linear projections for Q, K, V
        # TODO: Initialize output projection

        # TODO: Initialize linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # TODO: Initialize output projection
        self.W_o = nn.Linear(d_model, d_model)


    def split_heads(self, x):
        """
        Split tensor into multiple heads.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Tensor with shape [batch, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, _ = x.size()

        # TODO: Reshape and transpose to split heads
        # TODO: Reshape and transpose to split heads
        # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        # (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        return x.transpose(1, 2)

        # raise NotImplementedError

    def combine_heads(self, x):
        """
        Combine multiple heads back into single tensor.

        Args:
            x: Input tensor [batch, num_heads, seq_len, d_k]

        Returns:
            Tensor with shape [batch, seq_len, d_model]
        """
        batch_size, _, seq_len, d_k = x.size()

        # TODO: Transpose and reshape to combine heads
        # TODO: Transpose and reshape to combine heads
        # (batch, num_heads, seq_len, d_k) -> (batch, seq_len, num_heads, d_k)
        assert d_k == self.d_k
        x = x.transpose(1, 2).contiguous()
        
        # (batch, seq_len, num_heads, d_k) -> (batch, seq_len, d_model)
        return x.view(batch_size, seq_len, self.d_model)

        # raise NotImplementedError

    def forward(self, query, key, value, mask=None):
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor [batch, seq_len_q, d_model]
            key: Key tensor [batch, seq_len_k, d_model]
            value: Value tensor [batch, seq_len_v, d_model]
            mask: Optional attention mask

        Returns:
            output: Attention output [batch, seq_len_q, d_model]
            attention_weights: Attention weights [batch, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)

        # TODO: Linear projections
        # TODO: Split heads
        # TODO: Apply scaled dot-product attention
        # TODO: Combine heads
        # TODO: Apply output projection
        # TODO: Linear projections
        # [batch, seq_len_q, d_model]
        Q = self.W_q(query)
        # [batch, seq_len_k, d_model]
        K = self.W_k(key)
        # [batch, seq_len_v, d_model]
        V = self.W_v(value)

        # TODO: Split heads
        # [batch, num_heads, seq_len_q, d_k]
        Q_split = self.split_heads(Q)
        # [batch, num_heads, seq_len_k, d_k]
        K_split = self.split_heads(K)
        # [batch, num_heads, seq_len_v, d_k]
        V_split = self.split_heads(V)

        # TODO: Apply scaled dot-product attention
        # Note: If mask is provided, it should be broadcastable
        # to [batch, num_heads, seq_len_q, seq_len_k]
        if mask is not None:
            # Add head dimension if mask is [batch, 1, seq_len_q, seq_len_k]
            # or [batch, 1, 1, seq_len_k]
            # No change needed if mask is already [batch, num_heads, ...]
            # For this assignment, a simple [batch, 1, seq_len, seq_len] mask
            # will broadcast correctly to [batch, num_heads, seq_len, seq_len]
            pass # Mask is broadcast automatically
            
        attention_output, attention_weights = scaled_dot_product_attention(
            Q_split, K_split, V_split, mask
        )
        # attention_output: [batch, num_heads, seq_len_q, d_k]
        # attention_weights: [batch, num_heads, seq_len_q, seq_len_k]

        # TODO: Combine heads
        # [batch, seq_len_q, d_model]
        combined_output = self.combine_heads(attention_output)

        # TODO: Apply output projection
        # [batch, seq_len_q, d_model]
        output = self.W_o(combined_output)

        return output, attention_weights

        # raise NotImplementedError


def create_causal_mask(seq_len, device=None):
    """
    Create causal mask to prevent attending to future positions.

    Args:
        seq_len: Sequence length
        device: Device to create tensor on

    Returns:
        Mask tensor [1, 1, seq_len, seq_len] lower triangular matrix
    """
    # Lower triangular matrix
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)