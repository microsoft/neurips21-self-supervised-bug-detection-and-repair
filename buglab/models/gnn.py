from typing import Any, Callable, Dict, List, Literal, NamedTuple, Optional, Union

import logging
import numpy as np
import torch
from ptgnn.baseneuralmodel import AbstractNeuralModel
from ptgnn.neuralmodels.gnn import (
    GnnOutput,
    GraphData,
    GraphNeuralNetwork,
    GraphNeuralNetworkModel,
    TensorizedGraphData,
)
from torch import nn

from buglab.models.basemodel import AbstractBugLabModel, RewriteModuleSelectionInformation
from buglab.models.rewrite_chooser_module import RewriteChooserInformation, RewriteChooserModule
from buglab.representations.data import BugLabData, HypergraphData
from buglab.representations.hypergraph import convert_buglab_sample_to_hypergraph

LOGGER = logging.getLogger(__name__)


class GnnModelTensorizedSample(NamedTuple):
    graph_data: Any

    # Bug localization
    rewritten_node_idx: Optional[int]

    # Bug repair
    text_rewrites_info: RewriteModuleSelectionInformation
    varswap_rewrites_info: RewriteModuleSelectionInformation
    argswap_rewrites_info: RewriteModuleSelectionInformation

    num_rewrite_locations_considered: int

    # Bug selection
    observed_rewrite_logprobs: Optional[List[float]]


class GnnBugLabModule(RewriteChooserModule):
    def __init__(self, gnn: GraphNeuralNetwork, use_all_gnn_layer_outputs: bool = False, **kwargs):
        super().__init__(entity_repr_size=gnn.output_node_state_dim, **kwargs)
        self._gnn = gnn

        self.__use_all_gnn_layer_outputs = use_all_gnn_layer_outputs
        if use_all_gnn_layer_outputs:
            self.__summarization_layer = nn.Linear(
                in_features=self._gnn.input_node_state_dim
                + sum(mpl.output_state_dimension for mpl in gnn.message_passing_layers),
                out_features=self._gnn.output_node_state_dim,
            )

    @property
    def use_all_gnn_layer_outputs(self):
        return self.__use_all_gnn_layer_outputs

    @property
    def gnn(self):
        return self._gnn

    def __compute_gnn_output(self, graph_data):
        gnn_output: GnnOutput = self._gnn(**graph_data, return_all_states=self.__use_all_gnn_layer_outputs)
        if self.__use_all_gnn_layer_outputs:
            gnn_output = gnn_output._replace(
                output_node_representations=self.__summarization_layer(gnn_output.output_node_representations)
            )

        return gnn_output

    def compute_rewrite_chooser_information(
        self,
        *,
        graph_data: Dict[str, Any],
        sample_has_bug: torch.Tensor,
        # Text repair
        text_rewrite_replacement_ids: torch.Tensor,
        text_rewrite_to_loc_group: torch.Tensor,
        # Var Misuse
        varswap_to_loc_group: torch.Tensor,
        # ArgSwap
        argswap_to_loc_group: torch.Tensor,
        # Other inputs that we don't consume:
        **kwargs,
    ) -> RewriteChooserInformation:
        gnn_output = self.__compute_gnn_output(graph_data)

        return RewriteChooserInformation(
            num_samples=sample_has_bug.shape[0],
            rewritable_loc_reprs=gnn_output.output_node_representations[
                gnn_output.node_idx_references["candidate_nodes"]
            ],
            rewritable_loc_to_sample_id=gnn_output.node_graph_idx_reference["candidate_nodes"],
            text_rewrite_loc_reprs=gnn_output.output_node_representations[
                gnn_output.node_idx_references["text_rewrite_node_idxs"]
            ],
            text_rewrite_replacement_ids=text_rewrite_replacement_ids,
            text_rewrite_to_loc_group=text_rewrite_to_loc_group,
            varswap_loc_reprs=gnn_output.output_node_representations[
                gnn_output.node_idx_references["varswap_node_idxs"]
            ],
            varswap_replacement_reprs=gnn_output.output_node_representations[
                gnn_output.node_idx_references["varswap_replacement_node_idxs"]
            ],
            varswap_to_loc_group=varswap_to_loc_group,
            argswap_loc_reprs=gnn_output.output_node_representations[
                gnn_output.node_idx_references["argswap_node_idxs"]
            ],
            argswap_replacement_reprs=gnn_output.output_node_representations[
                gnn_output.node_idx_references["argswap_replacement_node_idxs"]
            ],
            argswap_to_loc_group=argswap_to_loc_group,
        )

    def forward(self, **kwargs):
        rc_info = self.compute_rewrite_chooser_information(**kwargs)
        return self.compute_loss(rc_info=rc_info, **kwargs)


class GnnBugLabModel(AbstractNeuralModel[BugLabData, GnnModelTensorizedSample, GnnBugLabModule], AbstractBugLabModel):
    def __init__(
        self,
        gnn_model: GraphNeuralNetworkModel,
        use_all_gnn_layer_outputs: bool = False,
        hyper=False,
        type_name_exclude_set=frozenset(),
        **kwargs,
    ):
        super().__init__()
        AbstractBugLabModel.__init__(self, **kwargs)
        self._init()
        self._gnn_model = gnn_model
        self._use_all_gnn_layer_outputs = use_all_gnn_layer_outputs
        self._hyper = hyper
        assert (
            len(type_name_exclude_set) == 0 or hyper
        ), "The `type_name_exclude_set` is ignored by non hypergraph models"
        self.__type_name_exclude_set = type_name_exclude_set

    @property
    def gnn_model(self):
        return self._gnn_model

    @property
    def use_all_gnn_layer_outputs(self):
        return self._use_all_gnn_layer_outputs

    def update_metadata_from(self, datapoint: BugLabData) -> None:
        if self._hyper:
            graph_data, _ = BugLabData.as_hypergraph_data(datapoint, self.__type_name_exclude_set)
        else:
            graph_data, _ = BugLabData.as_graph_data(datapoint)
        self._gnn_model.update_metadata_from(graph_data)

    def build_neural_module(self) -> GnnBugLabModule:
        return GnnBugLabModule(
            self._gnn_model.build_neural_module(),
            rewrite_vocabulary_size=len(self._target_rewrite_ops),
            use_all_gnn_layer_outputs=self._use_all_gnn_layer_outputs,
            generator_loss_type=self._generator_loss_type,
            buggy_samples_weight_schedule=self._buggy_samples_weight_schedule,
            repair_weight_schedule=self._repair_weight_schedule,
            localization_module_type=self._localization_module_type,
        )

    def _compute_reference_nodes(self, datapoint: BugLabData, graph_data: Union[GraphData, HypergraphData]):

        (
            text_rewrite_selection_info,
            varswap_rewrite_selection_info,
            argswap_rewrite_selection_info,
            num_rewrite_locations,
        ) = self._compute_rewrite_data(datapoint, graph_data.reference_nodes["candidate_nodes"])

        graph_data.reference_nodes["text_rewrite_node_idxs"] = text_rewrite_selection_info.node_idxs
        graph_data.reference_nodes["varswap_node_idxs"] = varswap_rewrite_selection_info.node_idxs
        graph_data.reference_nodes["varswap_replacement_node_idxs"] = varswap_rewrite_selection_info.replacement_ids
        graph_data.reference_nodes["argswap_node_idxs"] = argswap_rewrite_selection_info.node_idxs
        if len(argswap_rewrite_selection_info.replacement_ids) == 0:
            argswap_replacement_node_idxs = np.zeros((0, 2), dtype=np.int32)
        else:
            argswap_replacement_node_idxs = argswap_rewrite_selection_info.replacement_ids
        graph_data.reference_nodes["argswap_replacement_node_idxs"] = argswap_replacement_node_idxs

        assert (1 if text_rewrite_selection_info.correct_choice_idx else 0) + (
            1 if varswap_rewrite_selection_info.correct_choice_idx else 0
        ) + (
            1 if argswap_rewrite_selection_info.correct_choice_idx else 0
        ) <= 1, "No more than one node should be correct."
        return (
            text_rewrite_selection_info,
            varswap_rewrite_selection_info,
            argswap_rewrite_selection_info,
            num_rewrite_locations,
            graph_data,
        )

    def tensorize(self, datapoint: BugLabData) -> Optional[GnnModelTensorizedSample]:
        if self._hyper:
            graph_data, rewritten_node_idx = BugLabData.as_hypergraph_data(datapoint)
        else:
            graph_data, rewritten_node_idx = BugLabData.as_graph_data(datapoint)

            graph_data: GraphData

        (
            text_rewrite_selection_info,
            varswap_rewrite_selection_info,
            argswap_rewrite_selection_info,
            num_rewrite_locations,
            graph_data,
        ) = self._compute_reference_nodes(datapoint, graph_data)

        tensorized_graph_data = self._gnn_model.tensorize(graph_data)
        if tensorized_graph_data is None:
            return None

        return GnnModelTensorizedSample(
            graph_data=tensorized_graph_data,
            # Bug localization
            rewritten_node_idx=rewritten_node_idx,
            # Bug repair
            text_rewrites_info=text_rewrite_selection_info,
            varswap_rewrites_info=varswap_rewrite_selection_info,
            argswap_rewrites_info=argswap_rewrite_selection_info,
            num_rewrite_locations_considered=num_rewrite_locations,
            # Rewrite logprobs
            observed_rewrite_logprobs=datapoint.get("candidate_rewrite_logprobs", None),
        )

    def initialize_minibatch(self) -> Dict[str, Any]:
        return {
            "graph_data": self._gnn_model.initialize_minibatch(),
            # Bug Localization
            "sample_has_bug": [],
            "sample_to_correct_loc_idx": [],
            "mb_num_target_nodes": 0,
            "mb_num_rewrite_candidates": 0,
            # Repair
            "mb_num_repair_groups": 0,
            # Text rewrites
            "text_rewrite_replacement_ids": [],
            "text_rewrite_to_loc_group": [],
            "text_rewrite_correct_idxs": [],
            "text_rewrite_original_idxs": [],
            "text_rewrite_joint_idxs": [],
            # Var Misuse
            "varswap_to_loc_group": [],
            "varswap_correct_idxs": [],
            "varswap_original_idxs": [],
            "varswap_joint_idxs": [],
            # Arg Swaps
            "argswap_to_loc_group": [],
            "argswap_correct_idxs": [],
            "argswap_original_idxs": [],
            "argswap_joint_idxs": [],
            # Rewrite logprobs
            "rewrite_to_sample_id": [],
            "observed_rewrite_logprobs": [],
            "no_bug_observed_rewrite_logprobs": [],
        }

    def extend_minibatch_with(
        self, tensorized_datapoint: GnnModelTensorizedSample, partial_minibatch: Dict[str, Any]
    ) -> bool:
        current_sample_idx = len(partial_minibatch["graph_data"]["num_nodes_per_graph"])
        keep_extending = self._gnn_model.extend_minibatch_with(
            tensorized_datapoint.graph_data, partial_minibatch["graph_data"]
        )
        if tensorized_datapoint.rewritten_node_idx is None:
            partial_minibatch["sample_has_bug"].append(False)
            partial_minibatch["sample_to_correct_loc_idx"].append(-10000)  # Marker for "not buggy"
        else:
            partial_minibatch["sample_has_bug"].append(True)
            partial_minibatch["sample_to_correct_loc_idx"].append(
                tensorized_datapoint.rewritten_node_idx + partial_minibatch["mb_num_target_nodes"]
            )
        partial_minibatch["mb_num_target_nodes"] += len(
            tensorized_datapoint.graph_data.reference_nodes["candidate_nodes"]
        )

        mb_rewrite_candidate_offset = partial_minibatch["mb_num_rewrite_candidates"]
        num_rewrite_candidates_in_datapoint = 0
        mb_num_repair_groups_offset = partial_minibatch["mb_num_repair_groups"]

        # Text rewrites
        if tensorized_datapoint.text_rewrites_info.correct_choice_idx is not None:
            partial_minibatch["text_rewrite_correct_idxs"].append(
                tensorized_datapoint.text_rewrites_info.correct_choice_idx
                + len(partial_minibatch["text_rewrite_to_loc_group"])
            )
        partial_minibatch["text_rewrite_replacement_ids"].extend(
            tensorized_datapoint.text_rewrites_info.replacement_ids
        )
        partial_minibatch["text_rewrite_to_loc_group"].extend(
            t + mb_num_repair_groups_offset for t in tensorized_datapoint.text_rewrites_info.loc_groups
        )
        partial_minibatch["text_rewrite_joint_idxs"].extend(
            t + mb_rewrite_candidate_offset for t in tensorized_datapoint.text_rewrites_info.original_idxs
        )
        num_rewrite_candidates_in_datapoint += len(tensorized_datapoint.text_rewrites_info.original_idxs)

        # Var Misuse
        if tensorized_datapoint.varswap_rewrites_info.correct_choice_idx is not None:
            partial_minibatch["varswap_correct_idxs"].append(
                tensorized_datapoint.varswap_rewrites_info.correct_choice_idx
                + len(partial_minibatch["varswap_to_loc_group"])
            )
        partial_minibatch["varswap_to_loc_group"].extend(
            t + mb_num_repair_groups_offset for t in tensorized_datapoint.varswap_rewrites_info.loc_groups
        )
        partial_minibatch["varswap_joint_idxs"].extend(
            t + mb_rewrite_candidate_offset for t in tensorized_datapoint.varswap_rewrites_info.original_idxs
        )
        num_rewrite_candidates_in_datapoint += len(tensorized_datapoint.varswap_rewrites_info.original_idxs)

        # Arg Swaps
        if tensorized_datapoint.argswap_rewrites_info.correct_choice_idx is not None:
            partial_minibatch["argswap_correct_idxs"].append(
                tensorized_datapoint.argswap_rewrites_info.correct_choice_idx
                + len(partial_minibatch["argswap_to_loc_group"])
            )
        partial_minibatch["argswap_to_loc_group"].extend(
            t + mb_num_repair_groups_offset for t in tensorized_datapoint.argswap_rewrites_info.loc_groups
        )
        partial_minibatch["argswap_joint_idxs"].extend(
            t + mb_rewrite_candidate_offset for t in tensorized_datapoint.argswap_rewrites_info.original_idxs
        )
        num_rewrite_candidates_in_datapoint += len(tensorized_datapoint.argswap_rewrites_info.original_idxs)
        partial_minibatch["mb_num_rewrite_candidates"] += num_rewrite_candidates_in_datapoint

        partial_minibatch["mb_num_repair_groups"] += tensorized_datapoint.num_rewrite_locations_considered
        # Visualization info
        partial_minibatch["text_rewrite_original_idxs"].append(tensorized_datapoint.text_rewrites_info.original_idxs)
        partial_minibatch["varswap_original_idxs"].append(tensorized_datapoint.varswap_rewrites_info.original_idxs)
        partial_minibatch["argswap_original_idxs"].append(tensorized_datapoint.argswap_rewrites_info.original_idxs)

        # Rewrite logprobs
        partial_minibatch["rewrite_to_sample_id"].extend([current_sample_idx] * num_rewrite_candidates_in_datapoint)
        logprobs = tensorized_datapoint.observed_rewrite_logprobs
        if logprobs is not None:
            partial_minibatch["observed_rewrite_logprobs"].extend(logprobs[:-1])
            partial_minibatch["no_bug_observed_rewrite_logprobs"].append(logprobs[-1])

        return keep_extending

    def finalize_minibatch(
        self, accumulated_minibatch_data: Dict[str, Any], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        graph_data = self._gnn_model.finalize_minibatch(accumulated_minibatch_data["graph_data"], device)

        minibatch = {
            "graph_data": graph_data,
            "sample_to_correct_loc_idx": torch.tensor(
                accumulated_minibatch_data["sample_to_correct_loc_idx"], dtype=torch.int64, device=device
            ),
            "sample_has_bug": torch.tensor(
                accumulated_minibatch_data["sample_has_bug"], dtype=torch.bool, device=device
            ),
            # Text rewrites
            "text_rewrite_replacement_ids": torch.tensor(
                accumulated_minibatch_data["text_rewrite_replacement_ids"], dtype=torch.int64, device=device
            ),
            "text_rewrite_to_loc_group": torch.tensor(
                accumulated_minibatch_data["text_rewrite_to_loc_group"], dtype=torch.int64, device=device
            ),
            "text_rewrite_correct_idxs": torch.tensor(
                accumulated_minibatch_data["text_rewrite_correct_idxs"], dtype=torch.int64, device=device
            ),
            "text_rewrite_joint_idxs": torch.tensor(
                accumulated_minibatch_data["text_rewrite_joint_idxs"], dtype=torch.int64, device=device
            ),
            # Var Misuse
            "varswap_to_loc_group": torch.tensor(
                accumulated_minibatch_data["varswap_to_loc_group"], dtype=torch.int64, device=device
            ),
            "varswap_correct_idxs": torch.tensor(
                accumulated_minibatch_data["varswap_correct_idxs"], dtype=torch.int64, device=device
            ),
            "varswap_joint_idxs": torch.tensor(
                accumulated_minibatch_data["varswap_joint_idxs"], dtype=torch.int64, device=device
            ),
            # Arg swaps
            "argswap_to_loc_group": torch.tensor(
                accumulated_minibatch_data["argswap_to_loc_group"], dtype=torch.int64, device=device
            ),
            "argswap_correct_idxs": torch.tensor(
                accumulated_minibatch_data["argswap_correct_idxs"], dtype=torch.int64, device=device
            ),
            "argswap_joint_idxs": torch.tensor(
                accumulated_minibatch_data["argswap_joint_idxs"], dtype=torch.int64, device=device
            ),
            # Visualisation info
            "text_rewrite_original_idxs": accumulated_minibatch_data["text_rewrite_original_idxs"],
            "varswap_original_idxs": accumulated_minibatch_data["varswap_original_idxs"],
            "argswap_original_idxs": accumulated_minibatch_data["argswap_original_idxs"],
            # Rewrite logprobs
            "rewrite_to_sample_id": torch.tensor(
                accumulated_minibatch_data["rewrite_to_sample_id"], dtype=torch.int64, device=device
            ),
        }

        # Rewrite logprobs
        if accumulated_minibatch_data["observed_rewrite_logprobs"]:
            observed_rewrite_logprobs = (
                accumulated_minibatch_data["observed_rewrite_logprobs"]
                + accumulated_minibatch_data["no_bug_observed_rewrite_logprobs"]
            )
            minibatch["observed_rewrite_logprobs"] = torch.tensor(
                observed_rewrite_logprobs, dtype=torch.float32, device=device
            )
        return minibatch
