from typing import Any, Callable, Dict, Literal, Optional, Set, Tuple, Union

import logging
import re
from functools import partial
from pathlib import Path
from ptgnn.baseneuralmodel import AbstractNeuralModel, ModuleWithMetrics
from ptgnn.neuralmodels.embeddings.strelementrepresentationmodel import StrElementRepresentationModel
from ptgnn.neuralmodels.gnn import GraphNeuralNetworkModel

from buglab.models.gnn import GnnBugLabModel
from buglab.models.gnnlayerdefs import (
    create_ggnn_mp_layers,
    create_hybrid_mp_layers,
    create_mlp_mp_layers,
    create_mlp_mp_layers_no_residual,
    create_sandwich_mlp_mp_layers,
    create_sandwich_mlp_mp_layers_no_residual,
)
from buglab.models.hypergnn import HyperGnnModel
from buglab.models.layers.HGNN import HGNNModel
from buglab.models.layers.hyperedge_message_passing import HyperedgeMessagePassingModel
from buglab.models.layers.hyperedge_transformer import HyperedgeTransformerModel
from buglab.models.layers.mixstrreprmodel import MixedStrElementRepresentationModel
from buglab.models.seqmodel import SeqBugLabModel

LOGGER = logging.getLogger(__name__)


def const_schedule(epoch_idx: int, const_weight: float) -> float:
    return const_weight


LINEAR_INTERPOLATION_REGEX = re.compile(
    r"""
        interpolate\(
            (?P<num_epochs>[0-9]+)                     # always match a number of epochs
            (?:,\s?(?P<init_weight>[0-9]*\.[0-9]+))?   # allow a middle argument for initial weight (defaults to 1.0)
            (?:,\s?(?P<final_weight>[0-9]*\.[0-9]+))?  # final argument for final weight (defaults to 0.0)
        \)""",
    re.VERBOSE,
)


def linear_interpolation(epoch_idx: int, num_epochs: int, init_weight: float, final_weight: float) -> float:
    if epoch_idx < num_epochs:
        return init_weight + (final_weight - init_weight) * (epoch_idx / num_epochs)

    return final_weight


def weight_schedule(weight_spec: Union[str, int, float]) -> Callable[[int], float]:
    """Return a (serializable) function with the appropriate schedule"""
    if isinstance(weight_spec, (int, float)):
        return partial(const_schedule, const_weight=weight_spec)

    match = LINEAR_INTERPOLATION_REGEX.match(weight_spec)
    if match:
        num_epochs = int(match.group("num_epochs"))
        init_weight = float(match.group("init_weight") or "1.0")
        final_weight = float(match.group("final_weight") or "0.0")
        return partial(linear_interpolation, num_epochs=num_epochs, init_weight=init_weight, final_weight=final_weight)

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
    repair_weight_spec: Union[str, int, float] = 1.0,
    edge_feature_size: int = 0,
    localization_module_type: Optional[
        Literal["CandidateQuery", "RewriteQuery", "CandidateAndRewriteQuery"]
    ] = "CandidateAndRewriteQuery",
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
        buggy_samples_weight_schedule=weight_schedule(buggy_samples_weight_spec),
        repair_weight_schedule=weight_schedule(repair_weight_spec),
        localization_module_type=localization_module_type,
    )


def hyper_gnn(
    *,
    edge_repr_type: Literal["message_passing", "transformer", "HGNN"] = "message_passing",
    use_all_gnn_layer_outputs: bool = False,
    hidden_state_size: int = 256,
    dropout_rate: float = 0.1,
    node_representations: Optional[Dict[str, Any]] = None,
    selector_loss_type="classify-max-loss",
    stop_extending_minibatch_after_num_nodes: int = 30000,
    max_nodes_per_graph: int = 35000,
    max_memory: int = 300_000,
    buggy_samples_weight_spec: Union[str, int, float] = 1.0,
    repair_weight_spec: Union[str, int, float] = 1.0,
    arg_name_feature_size: int = 256,
    localization_module_type: Optional[
        Literal["CandidateQuery", "RewriteQuery", "CandidateAndRewriteQuery"]
    ] = "CandidateAndRewriteQuery",
    type_name_exclude_set: Set[str] = frozenset(),
    tie_weights: bool = False,
    norm_first: bool = False,
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

    hyperedge_type_representation_model = StrElementRepresentationModel(
        token_splitting="subtoken", embedding_size=hidden_state_size, dropout_rate=dropout_rate
    )
    base_hyperedge_argname_representation_model = StrElementRepresentationModel(
        token_splitting="subtoken",
        embedding_size=arg_name_feature_size,
        dropout_rate=dropout_rate,
    )
    hyperedge_argname_representation_model = MixedStrElementRepresentationModel(
        base_embedder_model=base_hyperedge_argname_representation_model,
    )

    if edge_repr_type == "message_passing":
        edge_representation_model = HyperedgeMessagePassingModel(
            message_input_state_size=hidden_state_size,
            hidden_state_size=hidden_state_size,
            argname_feature_state_size=arg_name_feature_size,
            dropout_rate=dropout_rate,
            num_layers=8,
            tie_weights=tie_weights,
            reduce_kind=kwargs.get("edge_reduce_kind", "max"),
        )
        node_update_type = "linear-residual"
        edge_update_type = None
    elif edge_repr_type == "transformer":
        edge_representation_model = HyperedgeTransformerModel(
            hidden_state_size=hidden_state_size,
            argname_feature_state_size=arg_name_feature_size,
            dropout_rate=dropout_rate,
            tie_weights=tie_weights,
            num_layers=6,
            dim_feedforward_transformer=2048,
            norm_first=norm_first,
        )
        node_update_type = edge_update_type = "transformer"
    elif edge_repr_type == "HGNN":
        edge_representation_model = HGNNModel(
            hidden_state_size=hidden_state_size,
            argname_feature_state_size=arg_name_feature_size,
            dropout_rate=dropout_rate,
            tie_weights=tie_weights,
            use_arg_names=kwargs.get("use_arg_names", False),
        )
        node_update_type = "linear"
        edge_update_type = None
    else:
        raise ValueError(f"Unknown edge representation type `{edge_repr_type}`!")

    return GnnBugLabModel(
        HyperGnnModel(
            node_representation_model=StrElementRepresentationModel(
                embedding_size=hidden_state_size, **node_representations
            ),
            hyperedge_arg_model=hyperedge_argname_representation_model,
            hyperedge_type_model=hyperedge_type_representation_model,
            max_nodes_per_graph=max_nodes_per_graph,
            max_memory=max_memory,
            hidden_state_size=hidden_state_size,
            stop_extending_minibatch_after_num_nodes=stop_extending_minibatch_after_num_nodes,
            node_update_type=node_update_type,
            edge_update_type=edge_update_type,
            hyperedge_representation_model=edge_representation_model,
            reduce_kind=kwargs.get("node_reduce_kind", "max"),
            edge_dropout_rate=0.0,
            dropout_rate=dropout_rate,
            normalise_by_node_degree=edge_repr_type == "HGNN",
        ),
        hyper=True,
        use_all_gnn_layer_outputs=use_all_gnn_layer_outputs,
        generator_loss_type=selector_loss_type,
        buggy_samples_weight_schedule=weight_schedule(buggy_samples_weight_spec),
        repair_weight_schedule=weight_schedule(repair_weight_spec),
        localization_module_type=localization_module_type,
        type_name_exclude_set=type_name_exclude_set,
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
    repair_weight_spec: Union[str, int, float] = 1.0,
    rezero_mode: Literal["off", "scalar", "vector"] = "off",
    normalisation_mode: Literal["off", "prenorm", "postnorm"] = "postnorm",
    localization_module_type: Literal[
        "CandidateQuery", "RewriteQuery", "CandidateAndRewriteQuery"
    ] = "CandidateAndRewriteQuery",
    **__,
):
    return SeqBugLabModel(
        hidden_state_size,
        max_subtoken_vocab_size=vocab_size,
        dropout_rate=dropout_rate,
        layer_type=layer_type,
        generator_loss_type=selector_loss_type,
        intermediate_dimension_size=intermediate_dimension_size,
        buggy_samples_weight_schedule=weight_schedule(buggy_samples_weight_spec),
        repair_weight_schedule=weight_schedule(repair_weight_spec),
        max_seq_size=max_seq_size,
        num_heads=num_heads,
        num_layers=num_layers,
        rezero_mode=rezero_mode,
        normalisation_mode=normalisation_mode,
        localization_module_type=localization_module_type,
    )


MODELS = {
    "gnn-mlp": lambda kwargs: gnn(mp_layer=create_mlp_mp_layers, add_self_edge=True, **kwargs),
    "hypergnn-mp": lambda kwargs: hyper_gnn(edge_repr_type="message_passing", **kwargs),
    "hypergnn-mp-residual": lambda kwargs: hyper_gnn(edge_repr_type="message_passing", **kwargs),
    "HGNN": lambda kwargs: hyper_gnn(arg_name_feature_size=32, edge_repr_type="HGNN", **kwargs),
    "hypergnn-transformer": lambda kwargs: hyper_gnn(edge_repr_type="transformer", **kwargs),
    "gnn-mlp-edge-attr": lambda kwargs: gnn(
        mp_layer=create_mlp_mp_layers, add_self_edge=True, edge_feature_size=8, **kwargs
    ),
    "gnn-sandwich": lambda kwargs: gnn(mp_layer=create_sandwich_mlp_mp_layers, add_self_edge=True, **kwargs),
    "ggnn": lambda kwargs: gnn(mp_layer=create_ggnn_mp_layers, add_self_edge=False, **kwargs),
    "seq-great": lambda kwargs: seq_transformer(layer_type="great", **kwargs),
    "seq-rat": lambda kwargs: seq_transformer(layer_type="rat", **kwargs),
    "seq-transformer": lambda kwargs: seq_transformer(layer_type="transformer", **kwargs),
    "seq-gru": lambda kwargs: seq_transformer(layer_type="gru", **kwargs),
}


def load_model(
    model_spec: Dict[str, Any],
    model_path: Path,
    restore_path: Optional[str] = None,
    restore_if_model_exists: bool = False,
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
        if model_spec["modelName"] not in MODELS:
            raise ValueError("Unknown model `%s`. Known models: %s", model_spec["modelName"], MODELS.keys())

        spec = dict(model_spec)
        del spec["modelName"]
        model = MODELS[model_spec["modelName"]](spec)
    return model, nn, initialize_metadata
