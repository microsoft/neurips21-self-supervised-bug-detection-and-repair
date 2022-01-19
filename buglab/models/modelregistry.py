import logging
import re
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

from ptgnn.baseneuralmodel import AbstractNeuralModel, ModuleWithMetrics
from ptgnn.neuralmodels.embeddings.strelementrepresentationmodel import StrElementRepresentationModel
from ptgnn.neuralmodels.gnn import GraphNeuralNetworkModel

from buglab.models.gnn import GnnBugLabModel
from buglab.models.gnnlayerdefs import create_ggnn_mp_layers, create_mlp_mp_layers
from buglab.models.seqmodel import SeqBugLabModel

LOGGER = logging.getLogger(__name__)


def const_schedule(epoch_idx: int, const_weight: float) -> float:
    return const_weight


WARMDOWN_WEIGHT_REGEX = re.compile("warmdown\\(([0-9]+),\\s?([0-9]*\\.[0-9]+)\\)")


def linear_warmdown(epoch_idx: int, num_warmdown_epochs: int, target_weight: float) -> float:
    return max(target_weight, epoch_idx * (target_weight - 1) / num_warmdown_epochs + 1)


def buggy_sample_weight_schedule(weight_spec: Union[str, int, float]) -> Callable[[int], float]:
    """Return a (serializable) function with the appropriate schedule"""
    if isinstance(weight_spec, (int, float)):
        return partial(const_schedule, const_weight=weight_spec)

    warmdown = WARMDOWN_WEIGHT_REGEX.match(weight_spec)
    if warmdown:
        num_warmdown_epochs = int(warmdown.group(1))
        target_weight = float(warmdown.group(2))
        # Linear decay up to the target
        return partial(linear_warmdown, num_warmdown_epochs=num_warmdown_epochs, target_weight=target_weight)

    raise Exception(f"Unrecognized buggy sample weighting `{weight_spec}`")


def gnn(
    *,
    mp_layer,
    add_self_edge: bool,
    use_all_gnn_layer_outputs: bool = False,
    hidden_state_size: int = 128,
    dropout_rate: float = 0.2,
    node_representations: Optional[Dict[str, Any]] = None,
    selector_loss_type="classify-max-loss",
    stop_extending_minibatch_after_num_nodes: int = 30000,
    max_nodes_per_graph: int = 35000,
    buggy_samples_weight_spec: Union[str, int, float] = 1.0,
    edge_feature_size: int = 0,
    **kwargs,
):
    if node_representations is None:
        node_representations = {}
    if "token_splitting" not in node_representations:
        node_representations["token_splitting"] = "subtoken"
    if "max_num_subtokens" not in node_representations:
        node_representations["max_num_subtokens"] = 6
    if "subtoken_combination" not in node_representations:
        node_representations["subtoken_combination"] = "max"
    if "vocabulary_size" not in node_representations:
        node_representations["vocabulary_size"] = 15000

    if edge_feature_size > 0:
        edge_representation_model = StrElementRepresentationModel(
            token_splitting="token",
            embedding_size=edge_feature_size,
        )
    else:
        edge_representation_model = None

    return GnnBugLabModel(
        GraphNeuralNetworkModel(
            node_representation_model=StrElementRepresentationModel(
                embedding_size=hidden_state_size, **node_representations
            ),
            edge_representation_model=edge_representation_model,
            add_self_edges=add_self_edge,
            message_passing_layer_creator=lambda n_edges: mp_layer(
                hidden_state_size, dropout_rate, n_edges, features_dimension=edge_feature_size
            ),
            stop_extending_minibatch_after_num_nodes=stop_extending_minibatch_after_num_nodes,
            max_nodes_per_graph=max_nodes_per_graph,
        ),
        use_all_gnn_layer_outputs=use_all_gnn_layer_outputs,
        generator_loss_type=selector_loss_type,
        buggy_samples_weight_schedule=buggy_sample_weight_schedule(buggy_samples_weight_spec),
    )


def seq_transformer(
    *,
    layer_type: Literal["great", "rat", "transformer", "gru"],
    hidden_state_size: int = 256,
    dropout_rate: float = 0.1,
    vocab_size: int = 15000,
    selector_loss_type: str = "classify-max-loss",
    num_layers: int = 5,
    num_heads: int = 8,
    max_seq_size: int = 400,
    intermediate_dimension_size: int = 1024,
    buggy_samples_weight_spec: Union[str, int, float] = 1.0,
    rezero_mode: Literal["off", "scalar", "vector"] = "off",
    normalisation_mode: Literal["off", "prenorm", "postnorm"] = "postnorm",
    **__,
):
    return SeqBugLabModel(
        hidden_state_size,
        max_subtoken_vocab_size=vocab_size,
        dropout_rate=dropout_rate,
        layer_type=layer_type,
        generator_loss_type=selector_loss_type,
        intermediate_dimension_size=intermediate_dimension_size,
        buggy_samples_weight_schedule=buggy_sample_weight_schedule(buggy_samples_weight_spec),
        max_seq_size=max_seq_size,
        num_heads=num_heads,
        num_layers=num_layers,
        rezero_mode=rezero_mode,
        normalisation_mode=normalisation_mode,
    )


def construct_model_dict(gnn_constructor: Callable, seq_constructor: Callable) -> Dict[str, Callable]:
    return {
        "gnn-mlp": lambda kwargs: gnn_constructor(mp_layer=create_mlp_mp_layers, add_self_edge=True, **kwargs),
        "ggnn": lambda kwargs: gnn_constructor(mp_layer=create_ggnn_mp_layers, add_self_edge=False, **kwargs),
        "seq-great": lambda kwargs: seq_constructor(layer_type="great", **kwargs),
        "seq-rat": lambda kwargs: seq_constructor(layer_type="rat", **kwargs),
        "seq-transformer": lambda kwargs: seq_constructor(layer_type="transformer", **kwargs),
        "seq-gru": lambda kwargs: seq_constructor(layer_type="gru", **kwargs),
    }


def load_model(
    model_spec: Dict[str, Any],
    model_path: Path,
    restore_path: Optional[str] = None,
    restore_if_model_exists: bool = False,
    type_model: bool = False,
) -> Tuple[AbstractNeuralModel, ModuleWithMetrics, bool]:
    assert model_path.name.endswith(".pkl.gz"), "MODEL_FILENAME must have a `.pkl.gz` suffix."
    initialize_metadata = True
    if restore_path is not None or (restore_if_model_exists and model_path.exists()):
        import torch

        LOGGER.info("Resuming training from %s." % model_path)
        initialize_metadata = False
        model, nn = AbstractNeuralModel.restore_model(
            model_path, torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
    else:
        nn = None
        models = construct_model_dict(gnn, seq_transformer)
        if model_spec["modelName"] not in models:
            raise ValueError("Unknown model `%s`. Known models: %s", model_spec["modelName"], models.keys())

        spec = dict(model_spec)
        del spec["modelName"]
        model = models[model_spec["modelName"]](spec)
    return model, nn, initialize_metadata
