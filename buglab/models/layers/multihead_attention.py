import math

import torch
from torch import nn


class MultiheadAttention(nn.Module):
    """
    A multihead attention implementation.
    """

    def __init__(
        self,
        *,
        num_heads: int,
        input_state_dimension: int,
        key_query_dimension: int,
        value_dimension: int,
        output_dimension: int,
        dropout_rate: float,
    ):
        super().__init__()

        self._num_heads = num_heads
        self._key_query_dim = key_query_dimension
        self._value_dim = value_dimension

        self._selfatt_head_transforms = nn.Linear(
            in_features=input_state_dimension,
            out_features=num_heads * (2 * key_query_dimension + value_dimension),
            bias=False,
        )

        self._dropout_layer = nn.Dropout(p=dropout_rate)
        self._scaling = self._key_query_dim ** -0.5
        self._out_proj = nn.Linear(value_dimension * num_heads, output_dimension, bias=False)

    def forward(self, input_seq_states, masked_elements):
        queries, keys, values = self._compute_qkv(input_seq_states)
        attention_scores = self._compute_attention_scores(keys, queries)
        attention_probs = self._compute_attention_probs(masked_elements, attention_scores)
        multiheaded_weighted_value_sum = self._compute_weighted_sum(values, attention_probs)
        return self._compute_output(multiheaded_weighted_value_sum)

    def _compute_qkv(self, input_seq_states):
        keys_queries_values = self._selfatt_head_transforms(input_seq_states).reshape(
            input_seq_states.shape[0], input_seq_states.shape[1], self._num_heads, -1
        )
        queries, keys, values = torch.split(
            keys_queries_values,
            split_size_or_sections=[self._key_query_dim, self._key_query_dim, self._value_dim],
            dim=-1,
        )  # [B, query_len, num_heads, key_dim], [B, memory_len, num_heads, key_dim], [B, memory_len, num_heads, value_dim]
        queries = queries * self._scaling

        return queries, keys, values

    def _compute_attention_scores(self, keys, queries):
        # Standard dot-attention: Here, we compute
        #    e_bijk = (in_bi * W_Q^k) * (in_bj * W_K^k)^T
        # i.e., the inner product of the query-projection of token in_bi and key-projection of token in_bj,
        # where b is the ID of the sample in the batch, i, j are token IDs, and k is the ID of a head.
        return torch.einsum("bkhd,bqhd->bqkh", keys, queries)  # [B, query_len, memory_len, num_heads]

    def _compute_attention_probs(self, masked_elements, raw_attention_scores):
        raw_attention_scores = raw_attention_scores.transpose(2, 3)  # [B, query_len, num_heads, memory_len]
        raw_attention_scores = raw_attention_scores
        if masked_elements is not None:
            raw_attention_scores.masked_fill_(masked_elements.unsqueeze(1).unsqueeze(1), -math.inf)

        attention_probs = nn.functional.softmax(raw_attention_scores, dim=-1)
        attention_probs = self._dropout_layer(attention_probs)  # Odd, but exists in original BERT

        return attention_probs

    def _compute_weighted_sum(self, values, attention_probs):
        return torch.einsum("blhq,bqhd->blhd", attention_probs, values)  # [B, query_len, num_heads, value_dim]

    def _compute_output(self, multiheaded_weighted_value_sum):
        attention_output = multiheaded_weighted_value_sum.reshape(
            (multiheaded_weighted_value_sum.shape[0], multiheaded_weighted_value_sum.shape[1], -1)
        )  # [B, query_len, num_heads * value_dim]
        attention_output = self._out_proj(attention_output)  # [B, query_len, out_dim]

        return attention_output
