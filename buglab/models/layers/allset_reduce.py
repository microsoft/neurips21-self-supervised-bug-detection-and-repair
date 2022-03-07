from typing_extensions import Literal

import math
import torch
import torch.nn as nn
from ptgnn.neuralmodels.reduceops import AbstractVarSizedElementReduce, ElementsToSummaryRepresentationInput
from torch_scatter import scatter, scatter_log_softmax, scatter_softmax, scatter_sum


class AllDeepSetElementReduce(AbstractVarSizedElementReduce):
    def __init__(
        self, hidden_state_size, summarization_type: Literal["sum", "mean", "max", "min"], dropout_rate: float = 0.0
    ):
        super().__init__()
        assert summarization_type in {"sum", "mean", "max", "min"}
        assert 0 <= dropout_rate < 1
        self.__summarization_type = summarization_type
        self.transform = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_state_size, hidden_state_size),
        )
        self.update = nn.Linear(hidden_state_size, hidden_state_size)

    def forward(self, inputs: ElementsToSummaryRepresentationInput):
        transformed = self.transform(inputs.element_embeddings)
        scattered = scatter(
            src=transformed,
            index=inputs.element_to_sample_map,
            dim=0,
            dim_size=inputs.num_samples,
            reduce=self.__summarization_type,
        )
        return self.update(scattered)


class AllSetTransformerReduce(AbstractVarSizedElementReduce):
    def __init__(
        self,
        hidden_size: int,
        dropout_rate: float = 0,
        num_heads: int = 8,
        use_value_layer: bool = True,
    ):
        input_representation_size = hidden_size
        output_representation_size = hidden_size
        super().__init__()
        self.__query_layer = nn.Linear(input_representation_size, num_heads, bias=False)
        self.__key_layer = nn.Linear(input_representation_size, hidden_size, bias=False)
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by the number of heads."
        self.__use_value_layer = use_value_layer
        if use_value_layer:
            self.__value_layer = nn.Linear(input_representation_size, hidden_size, bias=False)
            self.__output_layer = nn.Linear(hidden_size, output_representation_size, bias=False)
        else:
            self.__output_layer = nn.Linear(
                input_representation_size * num_heads, output_representation_size, bias=False
            )
        self.__num_heads = num_heads
        self.__layer_norm_hidden = nn.LayerNorm(hidden_size)
        self.__layer_norm_output = nn.LayerNorm(hidden_size)
        self.__queries = nn.Parameter(torch.randn(hidden_size))
        assert 0 <= dropout_rate < 1
        self.__dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs: ElementsToSummaryRepresentationInput) -> torch.Tensor:
        inputs = ElementsToSummaryRepresentationInput(
            element_embeddings=self.__dropout(inputs.element_embeddings),
            element_to_sample_map=inputs.element_to_sample_map,
            num_samples=inputs.num_samples,
        )

        queries = self.__queries.unsqueeze(0).expand(inputs.element_embeddings.shape[0], -1)  # [num_elements, H]
        queries = queries.reshape(
            (
                queries.shape[0],
                self.__num_heads,
                queries.shape[1] // self.__num_heads,
            )
        )

        keys = self.__key_layer(inputs.element_embeddings)  # [num_elements, H]
        keys = keys.reshape((keys.shape[0], self.__num_heads, keys.shape[1] // self.__num_heads))

        attention_scores = torch.einsum("bhk,bhk->bh", queries, keys)  # [num_elements, num_heads]
        attention_probs = torch.exp(
            scatter_log_softmax(attention_scores, index=inputs.element_to_sample_map, dim=0, eps=0)
        )  # [num_elements, num_heads]

        if self.__use_value_layer:
            values = self.__value_layer(inputs.element_embeddings)  # [num_elements, hidden_size]
            values = values.reshape((values.shape[0], self.__num_heads, values.shape[1] // self.__num_heads))
            outputs_per_head = attention_probs.unsqueeze(-1) * values
        else:
            outputs_per_head = attention_probs.unsqueeze(-1) * inputs.element_embeddings.unsqueeze(
                1
            )  # [num_elements, num_heads, D']

        outputs_multihead = outputs_per_head.reshape((outputs_per_head.shape[0], -1))  # [num_elements, num_heads * D']

        per_sample_outputs = scatter_sum(
            outputs_multihead, index=inputs.element_to_sample_map, dim=0, dim_size=inputs.num_samples
        )  # [num_samples, num_heads * D']
        per_sample_outputs = per_sample_outputs + self.__queries.unsqueeze(0)
        per_sample_outputs = self.__layer_norm_hidden(per_sample_outputs)  # [num_samples, num_heads * D']

        return self.__layer_norm_output(
            self.__output_layer(per_sample_outputs) + per_sample_outputs
        )  # [num_samples, num_heads * D']


class MultiheadSelfAttentionVarSizedKeysElementReduce(nn.Module):
    def __init__(
        self,
        input_representation_size: int,
        hidden_size: int,
        output_representation_size: int,
        num_heads: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.__query_layer = nn.Linear(input_representation_size, hidden_size, bias=True)

        self.__key_value_layer = nn.Linear(input_representation_size, 2 * hidden_size, bias=True)
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by the number of heads."

        self.__attention_scale = nn.Parameter(
            torch.tensor(1 / math.sqrt(hidden_size // num_heads)), requires_grad=False
        )
        self.__output_layer = nn.Linear(hidden_size, output_representation_size, bias=True)
        self.__dropout = nn.Dropout(dropout_rate)

        self.__num_heads = num_heads

    def forward(self, inputs: ElementsToSummaryRepresentationInput, queries) -> torch.Tensor:
        queries = self.__query_layer(queries)  # [num_samples, H]
        queries_per_element = queries[inputs.element_to_sample_map]  # [num_elements, H]
        queries_per_element = queries_per_element.reshape(
            (
                queries_per_element.shape[0],
                self.__num_heads,
                queries_per_element.shape[1] // self.__num_heads,
            )
        )  # [num_elements, num_heads, H // num_heads]

        key_values = self.__key_value_layer(inputs.element_embeddings)  # [num_elements, 2*H]
        key_values = key_values.reshape(
            (key_values.shape[0], 2, self.__num_heads, key_values.shape[1] // (self.__num_heads * 2))
        )
        keys, values = key_values[:, 0], key_values[:, 1]

        attention_scores = torch.einsum(
            "bkh,bkh,->bk", queries_per_element, keys, self.__attention_scale
        )  # [num_elements, num_heads]
        attention_probs = scatter_softmax(
            attention_scores, index=inputs.element_to_sample_map, dim=0, dim_size=queries.shape[0]
        )  # [num_elements, num_heads]
        attention_probs = self.__dropout(attention_probs)

        outputs = attention_probs.unsqueeze(-1) * values
        outputs = outputs.reshape((outputs.shape[0], -1))  # [num_elements, num_heads * D']

        per_sample_outputs = scatter_sum(
            outputs, index=inputs.element_to_sample_map, dim=0, dim_size=inputs.num_samples
        )  # [num_samples, num_heads, D']

        return self.__output_layer(per_sample_outputs)  # [num_samples, D']
