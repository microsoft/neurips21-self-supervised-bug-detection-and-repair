from typing import Literal, Optional, Union

import torch
from torch import nn

from buglab.models.layers.relational_multihead_attention import RelationalMultiheadAttention


def _get_activation_fn(activation):
    if activation == "relu":
        return nn.functional.relu
    elif activation == "gelu":
        return nn.functional.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class RelationalTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        key_query_dimension: int,
        value_dimension: int,
        nhead: int,
        num_edge_types: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation="relu",
        use_edge_value_biases: bool = False,
        edge_attention_bias_is_scalar: bool = False,
        rezero_mode: Literal["off", "scalar", "vector"] = "off",
        normalisation_mode: Literal["off", "prenorm", "postnorm"] = "postnorm",
    ):
        """
        Args:
            - rezero_mode: Three different modes are supported:
                * "off": No ReZero use.
                * "scalar": Sublayers (attention / fully connected) are scaled by a single scalar, i.e.,
                  \alpha is a scalar in the following:
                    x' = x + \alpha * SelfAtt(x)
                    x'' = x' + \alpha * Boom(x')
                    return x''
                  See https://arxiv.org/pdf/2003.04887.pdf.
                * "vector": Sublayers (attention / fully connected) are scaled by one value per dim, i.e.,
                  \alpha is a vector in the following:
                    x' = x + \alpha * SelfAtt(x)
                    x'' = x' + \alpha * Boom(x')
                    return x''
                  See https://arxiv.org/pdf/2103.17239.pdf.
            - normalisation_mode: Three different modes are supported:
                * "off": use no layer norm at all. Likely to diverge without using rezero as well.
                * "prenorm": Normalise values before each sublayer (attention / fully connected):
                    x' = x + SelfAtt(LN(x))
                    x'' = x' + Boom(LN(x'))
                    return x''
                * "postnorm": Normalise values after each sublayer:
                    x' = LN(x + SelfAtt(x))
                    x'' = LN(x' + Boom(x))
                    return x''
        """
        super(RelationalTransformerEncoderLayer, self).__init__()
        self.self_attn = RelationalMultiheadAttention(
            input_state_dimension=d_model,
            num_heads=nhead,
            output_dimension=d_model,
            dropout_rate=dropout,
            num_edge_types=num_edge_types,
            key_query_dimension=key_query_dimension,
            value_dimension=value_dimension,
            use_edge_value_biases=use_edge_value_biases,
            edge_attention_bias_is_scalar=edge_attention_bias_is_scalar,
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self._normalisation_mode = normalisation_mode
        if normalisation_mode in ("prenorm", "postnorm"):
            self.norm1: Optional[nn.LayerNorm] = nn.LayerNorm(d_model)
            self.norm2: Optional[nn.LayerNorm] = nn.LayerNorm(d_model)
        elif normalisation_mode == "off":
            self.norm1 = None
            self.norm2 = None
        else:
            raise ValueError(f"Unrecognized normalization mode `{normalisation_mode}`.")

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self._rezero_mode = rezero_mode
        if rezero_mode == "off":
            self._alpha1: Union[float, torch.Tensor] = 1.0
            self._alpha2: Union[float, torch.Tensor] = 1.0
        elif rezero_mode == "scalar":
            self._alpha1 = nn.Parameter(torch.tensor(0.0))
            self._alpha2 = nn.Parameter(torch.tensor(0.0))
        elif rezero_mode == "vector":
            self._alpha1 = nn.Parameter(torch.zeros(size=(d_model,)))
            self._alpha2 = nn.Parameter(torch.zeros(size=(d_model,)))
        else:
            raise ValueError(f"Unrecognized rezero mode `{rezero_mode}`.")

    def forward(self, src, src_mask, edges, edge_types):
        # --- Sublayer 1: Self-Attention:
        attn_input = src
        if self._normalisation_mode == "prenorm":
            attn_input = self.norm1(src)
        src2 = self.self_attn(attn_input, src_mask, edges, edge_types)
        src2 = self._alpha1 * src2
        src = src + self.dropout1(src2)
        if self._normalisation_mode == "postnorm":
            src = self.norm1(src)

        fc_input = src
        if self._normalisation_mode == "prenorm":
            fc_input = self.norm2(fc_input)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(fc_input))))
        src2 = self._alpha2 * src2
        src = src + self.dropout2(src2)
        if self._normalisation_mode == "postnorm":
            src = self.norm2(src)
        return src
