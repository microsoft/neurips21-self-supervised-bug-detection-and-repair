import torch
from torch import nn

from buglab.models.layers.multihead_attention import MultiheadAttention


class RelationalMultiheadAttention(MultiheadAttention):
    """
    A relational multihead implementation supporting two variations of using additional
    relationship information between tokens:

    * If no edge information is passed in in .forward(), this behaves like a standard
      multi-head self-attention.
    * If edges are present and edge_attention_bias_is_scalar=False,
      and use_edge_value_biases=True is set, this implements
          Eqs. (3) and (4) of
            Shaw, Peter, Jakob Uszkoreit, and Ashish Vaswani. "Self-attention with relative position representations."
            In ACL 2018.  https://www.aclweb.org/anthology/N18-2074/
        and
          Eq. (2) of
            Wang, Bailin, et al. "RAT-SQL: Relation-aware schema encoding and linking for text-to-SQL parsers."
            In ICML 2020.  https://arxiv.org/pdf/1911.04942.pdf
    * If edges are present and edge_attention_bias_is_scalar=True,
      and use_edge_value_biases=False is set, this implements Sect. 3.1 of
        Hellendoorn, Vincent J., et al. "Global relational models of source code."
        In ICLR 2020. https://openreview.net/pdf?id=B1lnbRNtwr
    """

    def __init__(
        self,
        *,
        num_heads: int,
        num_edge_types: int,
        input_state_dimension: int,
        key_query_dimension: int,
        value_dimension: int,
        output_dimension: int,
        dropout_rate: float,
        use_edge_value_biases: bool = False,
        edge_attention_bias_is_scalar: bool = False,
    ):
        super().__init__(
            num_heads=num_heads,
            input_state_dimension=input_state_dimension,
            key_query_dimension=key_query_dimension,
            value_dimension=value_dimension,
            output_dimension=output_dimension,
            dropout_rate=dropout_rate,
        )

        self._use_edge_value_biases = use_edge_value_biases
        self._edge_attention_bias_is_scalar = edge_attention_bias_is_scalar

        if self._edge_attention_bias_is_scalar:
            edge_attention_bias_dim = num_heads
        else:
            edge_attention_bias_dim = num_heads * key_query_dimension
        self._edge_attention_biases = nn.Embedding(num_embeddings=num_edge_types, embedding_dim=edge_attention_bias_dim)
        self._reverse_edge_attention_biases = nn.Embedding(
            num_embeddings=num_edge_types, embedding_dim=edge_attention_bias_dim
        )

        if self._use_edge_value_biases:
            self._edge_value_biases = nn.Embedding(
                num_embeddings=num_edge_types, embedding_dim=num_heads * value_dimension
            )
            self._reverse_edge_value_biases = nn.Embedding(
                num_embeddings=num_edge_types, embedding_dim=num_heads * value_dimension
            )

    def forward(self, input_seq_states, masked_elements, edges, edge_types):
        edge_sample_ids = edges[:, 0]
        edge_sources = edges[:, 1]
        edge_targets = edges[:, 2]

        queries, keys, values = self._compute_qkv(input_seq_states)
        raw_attention_scores = self._compute_attention_scores(keys, queries)
        attention_scores = self._add_edge_attention_scores(
            edge_sample_ids, edge_sources, edge_targets, edge_types, keys, queries, raw_attention_scores
        )
        attention_probs = self._compute_attention_probs(masked_elements, attention_scores)
        multiheaded_weighted_value_sum = self._compute_weighted_sum(values, attention_probs)
        if self._use_edge_value_biases:
            multiheaded_weighted_value_sum = self._add_edge_value_biases(
                edge_sample_ids, edge_sources, edge_targets, edge_types, attention_probs, multiheaded_weighted_value_sum
            )
        return self._compute_output(multiheaded_weighted_value_sum)

    def _add_edge_attention_scores(
        self, edge_sample_ids, edge_sources, edge_targets, edge_types, keys, queries, raw_attention_scores
    ):
        # We compute (sparse, per existing edge) additional bias scores e'_bijk:
        edge_bias_scores = self._compute_edge_bias_scores(
            edge_sample_ids, edge_sources, edge_targets, edge_types, keys, queries
        )

        # We add the e'_bijk (where present) to e_bijk. This should be a simple +=, but
        # that doesn't accumulate if we have several entries to add to e_bij. Hence we use
        # index_put_, which in turn requires a contiguous Tensor memory layout, and so we need
        # to establish that first:
        attention_scores = raw_attention_scores.contiguous()
        edge_sample_indices = torch.cat([edge_sample_ids, edge_sample_ids])
        edge_query_indices = torch.cat([edge_sources, edge_targets])
        edge_key_indices = torch.cat([edge_targets, edge_sources])
        attention_scores.index_put_(
            indices=(edge_sample_indices, edge_query_indices, edge_key_indices),
            values=edge_bias_scores,
            accumulate=True,
        )

        return attention_scores

    def _compute_edge_bias_scores(self, edge_sample_ids, edge_sources, edge_targets, edge_types, keys, queries):
        # We will compute additional e'_bihj which will be added onto the standard attention scores:
        attention_biases = self._edge_attention_biases(edge_types)
        attention_biases_r = self._reverse_edge_attention_biases(edge_types)

        if self._edge_attention_bias_is_scalar:
            # Compute e'_bijk = \sum_d (bias_bijk * (in_bj * W_K^k))_d
            # This is the GREAT model. Note two things:
            #  (1) This is defined on the _key_ representation, not the _query_ repr.
            #  (2) Because bias_bijk is a scalar, this is essentially just scaling
            #      (in_bj * W_K^k) and then summing.
            edge_attention_scores = torch.einsum(
                "eh,ehd->eh",
                attention_biases,  # [num_edges, num_heads]
                keys[edge_sample_ids, edge_targets],  # [num_edges, num_heads, key_dim]
            )  # [num_edges, num_head]
            r_edge_attention_scores = torch.einsum(
                "eh,ehd->eh",
                attention_biases_r,  # [num_edges, num_heads]
                keys[edge_sample_ids, edge_sources],  # [num_edges, num_heads, key_dim]
            )  # [num_edges, num_head]
            edge_bias_scores = torch.cat([edge_attention_scores, r_edge_attention_scores])  # [2 * num_edges, num_head]
        else:
            # Compute e'_bijk = (in_bj * W_Q^k) * bias_bijk^T
            # This is the Relative Position Representations / RAT-SQL variant. Note that this
            # is defined using the query representation, not the key repr.
            edge_attention_scores = torch.einsum(
                "ehd,ehd->eh",
                attention_biases.reshape((-1, self._num_heads, self._key_query_dim)),
                # [num_edges, num_heads, key_dim]
                queries[edge_sample_ids, edge_sources],  # [num_edges, num_heads, key_dim]
            )  # [num_edges, num_head]
            r_edge_attention_scores = torch.einsum(
                "ehd,ehd->eh",
                attention_biases_r.reshape((-1, self._num_heads, self._key_query_dim)),
                # [num_edges, num_heads, key_dim]
                queries[edge_sample_ids, edge_targets],  # [num_edges, num_heads, key_dim]
            )  # [num_edges, num_head]
            edge_bias_scores = torch.cat([edge_attention_scores, r_edge_attention_scores])  # [2 * num_edges, num_head]
        return edge_bias_scores

    def _add_edge_value_biases(
        self, edge_sample_ids, edge_sources, edge_targets, edge_types, attention_probs, multiheaded_weighted_value_sum
    ):
        edge_sample_indices = torch.cat([edge_sample_ids, edge_sample_ids])
        edge_query_indices = torch.cat([edge_sources, edge_targets])

        value_biases_shape = (edge_sample_ids.shape[0], self._num_heads, self._value_dim)
        value_bias_per_edge = attention_probs[edge_sample_ids, edge_sources, :, edge_targets].unsqueeze(
            -1
        ) * self._edge_value_biases(edge_types).reshape(
            value_biases_shape
        )  # [num_edges, num_heads, value_dim]
        value_bias_per_r_edge = attention_probs[edge_sample_ids, edge_targets, :, edge_sources].unsqueeze(
            -1
        ) * self._reverse_edge_value_biases(edge_types).reshape(
            value_biases_shape
        )  # [num_edges, num_heads, value_dim]

        biased_weighted_value_sum = multiheaded_weighted_value_sum.contiguous()
        biased_weighted_value_sum.index_put_(
            indices=(edge_sample_indices, edge_query_indices),
            values=torch.cat((value_bias_per_edge, value_bias_per_r_edge), dim=0),
            accumulate=True,
        )
        return biased_weighted_value_sum
