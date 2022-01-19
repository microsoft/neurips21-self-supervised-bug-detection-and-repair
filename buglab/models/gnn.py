import logging
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
from ptgnn.baseneuralmodel import AbstractNeuralModel, ModuleWithMetrics
from ptgnn.neuralmodels.gnn import (
    GnnOutput,
    GraphData,
    GraphNeuralNetwork,
    GraphNeuralNetworkModel,
    TensorizedGraphData,
)
from torch import nn

from buglab.models.basemodel import AbstractBugLabModel
from buglab.models.layers.fixermodules import (
    CandidatePairSelectorModule,
    SingleCandidateNodeSelectorModule,
    TextRepairModule,
)
from buglab.models.layers.localizationmodule import LocalizationModule
from buglab.models.utils import compute_generator_loss, scatter_log_softmax, scatter_max
from buglab.representations.data import BugLabData

LOGGER = logging.getLogger(__name__)


class BaseTensorizedBugLabGnn(NamedTuple):
    graph_data: TensorizedGraphData

    # Bug localization
    target_location_node_idx: Optional[int]

    # Bug repair
    target_rewrites: List[int]
    target_rewrite_to_location_group: List[int]
    correct_rewrite_target: Optional[int]
    text_rewrite_original_idx: List[int]

    candidate_symbol_to_varmisused_node: List[int]
    correct_candidate_symbol_node: Optional[int]
    candidate_rewrite_original_idx: List[int]

    swapped_pair_to_call: List[int]
    correct_swapped_pair: Optional[int]
    pair_rewrite_original_idx: List[int]

    num_rewrite_locations_considered: int

    # Bug selection
    rewrite_logprobs: Optional[List[float]]


class GnnBugLabModule(ModuleWithMetrics):
    def __init__(
        self,
        gnn: GraphNeuralNetwork,
        rewrite_vocabulary_size: int,
        use_all_gnn_layer_outputs: bool = False,
        generator_loss_type: Optional[str] = "norm-kl",
        buggy_samples_weight_schedule: Callable[[int], float] = lambda _: 1.0,
    ):
        super().__init__()
        self.__generator_loss_type = generator_loss_type
        self._gnn = gnn

        self.__use_all_gnn_layer_outputs = use_all_gnn_layer_outputs
        if use_all_gnn_layer_outputs:
            self.__summarization_layer = nn.Linear(
                in_features=self._gnn.input_node_state_dim
                + sum(mpl.output_state_dimension for mpl in gnn.message_passing_layers),
                out_features=self._gnn.output_node_state_dim,
            )

        output_state_dim = self._gnn.output_node_state_dim

        self.__localization_module = LocalizationModule(
            output_state_dim, buggy_samples_weight_schedule=buggy_samples_weight_schedule
        )
        self._buggy_samples_weight_schedule = buggy_samples_weight_schedule

        self._text_repair_module = TextRepairModule(output_state_dim, rewrite_vocabulary_size)
        self._varmisuse_module = SingleCandidateNodeSelectorModule(output_state_dim)
        self._argswap_module = CandidatePairSelectorModule(output_state_dim)

    @property
    def use_all_gnn_layer_outputs(self):
        return self.__use_all_gnn_layer_outputs

    @property
    def gnn(self):
        return self._gnn

    def _reset_module_metrics(self) -> None:
        if not hasattr(self, "_epoch_idx"):
            self._epoch_idx = 0
        elif self.training and self.__num_batches > 0:
            # Assumes that module metrics are reset once per epoch.
            self._epoch_idx += 1

        self.__loss = 0.0
        self.__repair_loss = 0.0

        self.__total_samples = 0
        self.__num_batches = 0

    def _module_metrics(self) -> Dict[str, Any]:
        metrics = {}
        if self.__total_samples > 0:
            metrics["Repair Loss"] = self.__repair_loss / self.__total_samples
        if self.__num_batches > 0:
            metrics["Loss"] = self.__loss / self.__num_batches
        return metrics

    def __compute_gnn_output(self, graph_data):
        gnn_output: GnnOutput = self._gnn(**graph_data, return_all_states=self.__use_all_gnn_layer_outputs)
        if self.__use_all_gnn_layer_outputs:
            gnn_output = gnn_output._replace(
                output_node_representations=self.__summarization_layer(gnn_output.output_node_representations)
            )

        return gnn_output

    def compute_localization_logprobs(self, graph_data: Dict[str, Any]):
        gnn_output = self.__compute_gnn_output(graph_data)

        candidate_node_reprs = gnn_output.output_node_representations[
            gnn_output.node_idx_references["candidate_nodes"]
        ]  # [C, H]

        (
            candidate_node_graph_idx,
            candidate_node_log_probs,
            arange,
        ) = self.__localization_module.compute_localization_logprobs(
            candidate_reprs=candidate_node_reprs,
            candidate_to_sample_idx=gnn_output.node_graph_idx_reference["candidate_nodes"],
            num_samples=gnn_output.num_graphs,
        )

        return candidate_node_graph_idx, candidate_node_log_probs, gnn_output, arange

    def forward(
        self,
        *,
        graph_data: Dict[str, Any],
        correct_candidate_node_idxs,
        has_bug: torch.Tensor,
        # Text repair
        target_rewrites: torch.IntTensor,
        rewrite_to_location_group: torch.IntTensor,
        correct_rewrite_idxs: torch.IntTensor,
        text_rewrite_idxs: torch.IntTensor,
        # Var Misuse
        candidate_symbol_to_location_group: torch.IntTensor,
        correct_candidate_symbols: torch.IntTensor,
        candidate_rewrite_idxs: torch.IntTensor,
        # ArgSwap
        swapped_pair_to_call_location_group: torch.IntTensor,
        correct_swapped_pair: torch.IntTensor,
        pair_rewrite_idxs: torch.IntTensor,
        # Rewrite logprobs
        rewrite_to_graph_id: torch.IntTensor,
        rewrite_logprobs: Optional[torch.FloatTensor] = None,
        **kwargs,  # Visualization data (ignored)
    ):
        gnn_output = self.__compute_gnn_output(graph_data)

        candidate_node_reprs = gnn_output.output_node_representations[
            gnn_output.node_idx_references["candidate_nodes"]
        ]  # [C, H]

        # Repair
        (
            arg_swap_logprobs,
            text_repair_logprobs,
            varmisuse_logprobs,
            (arg_swap_is_selected_fix, text_repair_is_selected_fix, varmisuse_is_selected_fix),
        ) = self._compute_repair_logprobs(
            gnn_output,
            target_rewrites,
            rewrite_to_location_group,
            candidate_symbol_to_location_group,
            swapped_pair_to_call_location_group,
        )

        # Calculate the loss for the generator:
        if rewrite_logprobs is not None:
            candidate_to_sample_idx = gnn_output.node_graph_idx_reference["candidate_nodes"]
            loss_type = self.__generator_loss_type
            _, localization_logprobs, arrange, = self.__localization_module.compute_localization_logprobs(
                candidate_reprs=candidate_node_reprs,
                candidate_to_sample_idx=candidate_to_sample_idx,
                num_samples=has_bug.shape[0],
            )

            loss = compute_generator_loss(
                arg_swap_logprobs,
                arrange,
                candidate_rewrite_idxs,
                candidate_symbol_to_location_group,
                localization_logprobs,
                loss_type,
                pair_rewrite_idxs,
                rewrite_logprobs,
                rewrite_to_graph_id,
                rewrite_to_location_group,
                swapped_pair_to_call_location_group,
                text_repair_logprobs,
                text_rewrite_idxs,
                varmisuse_logprobs,
            )

            with torch.no_grad():
                self.__loss += float(loss)
                self.__num_batches += 1

            return loss

        else:
            # Calculate the loss for the discriminator:
            localization_loss = self.__localization_module(
                candidate_reprs=candidate_node_reprs,
                candidate_to_sample_idx=gnn_output.node_graph_idx_reference["candidate_nodes"],
                has_bug=has_bug,
                correct_candidate_idxs=correct_candidate_node_idxs,
            )

            text_repair_loss = self._text_repair_module(
                text_repair_logprobs, correct_rewrite_idxs, selected_fixes=text_repair_is_selected_fix
            )
            varmisuse_repair_loss = self._varmisuse_module(
                varmisuse_logprobs, correct_candidate_symbols, selected_fixes=varmisuse_is_selected_fix
            )
            arg_swap_repair_loss = self._argswap_module(
                arg_swap_logprobs, correct_swapped_pair, selected_fixes=arg_swap_is_selected_fix
            )

            buggy_samples_weight = self._buggy_samples_weight_schedule(self._epoch_idx)
            repair_loss = text_repair_loss.sum() + varmisuse_repair_loss.sum() + arg_swap_repair_loss.sum()
            repair_loss = repair_loss * buggy_samples_weight

            with torch.no_grad():
                self.__total_samples += float(has_bug.sum().cpu())
                self.__repair_loss += float(repair_loss)

                self.__num_batches += 1
                self.__loss += float(localization_loss + repair_loss / has_bug.shape[0])

            return localization_loss + repair_loss / has_bug.shape[0]

    def _compute_repair_logprobs(
        self,
        gnn_output,
        target_rewrites,
        rewrite_to_location_group,
        candidate_symbol_to_location_group,
        swapped_pair_to_call_location_group,
    ):
        if target_rewrites.shape[0] > 0:
            text_repair_logits = self._text_repair_module.compute_rewrite_logits(
                target_rewrite_node_representations=gnn_output.output_node_representations[
                    gnn_output.node_idx_references["target_rewrite_nodes"]
                ],
                candidate_rewrites=target_rewrites,
            )
        else:
            text_repair_logits = torch.zeros(0, device=target_rewrites.device)

        if gnn_output.node_idx_references["varmisused_node_ids"].shape[0] > 0:
            varmisuse_logits = self._varmisuse_module.compute_per_slot_log_probability(
                slot_representations_per_target=gnn_output.output_node_representations[
                    gnn_output.node_idx_references["varmisused_node_ids"]
                ],
                target_nodes_representations=gnn_output.output_node_representations[
                    gnn_output.node_idx_references["candidate_symbol_node_ids"]
                ],
            )
        else:
            varmisuse_logits = torch.zeros(0, device=gnn_output.node_idx_references["varmisused_node_ids"].device)

        if gnn_output.node_idx_references["call_node_ids"].shape[0] > 0:
            arg_swap_logits = self._argswap_module.compute_per_pair_logits(
                slot_representations_per_pair=gnn_output.output_node_representations[
                    gnn_output.node_idx_references["call_node_ids"]
                ],
                pair_representations=gnn_output.output_node_representations[
                    gnn_output.node_idx_references["candidate_swapped_node_ids"]
                ],
            )
        else:
            arg_swap_logits = torch.zeros(0, device=gnn_output.node_idx_references["call_node_ids"].device)

        all_logits = torch.cat((text_repair_logits, varmisuse_logits, arg_swap_logits))
        logit_groups = torch.cat(
            (rewrite_to_location_group, candidate_symbol_to_location_group, swapped_pair_to_call_location_group)
        )
        logprobs = scatter_log_softmax(all_logits, index=logit_groups)
        text_repair_logprobs, varmisuse_logprobs, arg_swap_logprobs = torch.split(
            logprobs, [text_repair_logits.shape[0], varmisuse_logits.shape[0], arg_swap_logits.shape[0]]
        )

        with torch.no_grad():
            max_logit_per_group = scatter_max(all_logits, logit_groups)[0]
            max_per_rewrite = max_logit_per_group.gather(-1, logit_groups)

            is_selected_fix = max_per_rewrite == all_logits
            text_repair_is_selected_fix, varmisuse_is_selected_fix, arg_swap_is_selected_fix = torch.split(
                is_selected_fix, [text_repair_logits.shape[0], varmisuse_logits.shape[0], arg_swap_logits.shape[0]]
            )

        return (
            arg_swap_logprobs,
            text_repair_logprobs,
            varmisuse_logprobs,
            (
                arg_swap_is_selected_fix,
                text_repair_is_selected_fix,
                varmisuse_is_selected_fix,
            ),
        )


class GnnBugLabModel(AbstractNeuralModel[BugLabData, BaseTensorizedBugLabGnn, GnnBugLabModule], AbstractBugLabModel):
    def __init__(
        self,
        gnn_model: GraphNeuralNetworkModel,
        use_all_gnn_layer_outputs: bool = False,
        generator_loss_type: Optional[str] = "classify-max-loss",
        buggy_samples_weight_schedule: Callable[[int], float] = lambda _: 1.0,
    ):
        super().__init__()
        self._init()
        self.__gnn_model = gnn_model
        self.__use_all_gnn_layer_outputs = use_all_gnn_layer_outputs
        self.__generator_loss_type = generator_loss_type
        self.__buggy_samples_weight_schedule = buggy_samples_weight_schedule

    @property
    def gnn_model(self):
        return self.__gnn_model

    @property
    def use_all_gnn_layer_outputs(self):
        return self.__use_all_gnn_layer_outputs

    def update_metadata_from(self, datapoint: BugLabData) -> None:
        graph_data, _ = BugLabData.as_graph_data(datapoint)
        self.__gnn_model.update_metadata_from(graph_data)

    def build_neural_module(self) -> GnnBugLabModule:
        return GnnBugLabModule(
            self.__gnn_model.build_neural_module(),
            rewrite_vocabulary_size=len(self._target_rewrite_ops),
            use_all_gnn_layer_outputs=self.__use_all_gnn_layer_outputs,
            generator_loss_type=self.__generator_loss_type,
            buggy_samples_weight_schedule=self.__buggy_samples_weight_schedule,
        )

    def tensorize(self, datapoint: BugLabData) -> Optional[BaseTensorizedBugLabGnn]:
        graph_data, target_location_node_idx = BugLabData.as_graph_data(datapoint)
        graph_data: GraphData

        if "candidate_rewrite_logprobs" in datapoint:
            assert not self._tensorize_only_at_target_location_rewrites

        (
            # Rewrites
            target_rewrite_node_ids,
            target_rewrites,
            target_rewrite_to_location_group,
            correct_rewrite_target,
            text_rewrite_original_idx,
            # VarMisuse
            varmisused_node_ids,
            candidate_symbol_to_varmisused_location,
            candidate_symbol_node_ids,
            correct_candidate_symbol_node,
            varmisuse_rewrite_original_idx,
            # Arg Swap
            call_node_ids,
            candidate_swapped_node_ids,
            correct_swapped_pair,
            swapped_pair_to_call,
            swapped_rewrite_original_ids,
            # All rewrites
            repr_location_group_ids,
        ) = self._compute_rewrite_data(datapoint, graph_data.reference_nodes["candidate_nodes"])

        graph_data.reference_nodes["target_rewrite_nodes"] = target_rewrite_node_ids
        graph_data.reference_nodes["varmisused_node_ids"] = varmisused_node_ids
        graph_data.reference_nodes["candidate_symbol_node_ids"] = candidate_symbol_node_ids
        graph_data.reference_nodes["call_node_ids"] = call_node_ids
        if len(candidate_swapped_node_ids) == 0:
            candidate_swapped_node_ids = np.zeros((0, 2), dtype=np.int32)
        graph_data.reference_nodes["candidate_swapped_node_ids"] = candidate_swapped_node_ids

        assert (1 if correct_rewrite_target else 0) + (1 if correct_candidate_symbol_node else 0) + (
            1 if correct_swapped_pair else 0
        ) <= 1, "No more than one node should be correct."

        tensorized_graph_data = self.__gnn_model.tensorize(graph_data)
        if tensorized_graph_data is None:
            return None

        return BaseTensorizedBugLabGnn(
            graph_data=tensorized_graph_data,
            # Bug localization
            target_location_node_idx=target_location_node_idx,
            ### Bug repair
            # Text rewrite
            target_rewrites=target_rewrites,
            target_rewrite_to_location_group=target_rewrite_to_location_group,
            correct_rewrite_target=correct_rewrite_target,
            text_rewrite_original_idx=text_rewrite_original_idx,
            # Var Misuse
            candidate_symbol_to_varmisused_node=candidate_symbol_to_varmisused_location,
            correct_candidate_symbol_node=correct_candidate_symbol_node,
            candidate_rewrite_original_idx=varmisuse_rewrite_original_idx,
            # Arg Swap
            swapped_pair_to_call=swapped_pair_to_call,
            correct_swapped_pair=correct_swapped_pair,
            pair_rewrite_original_idx=swapped_rewrite_original_ids,
            # All rewrites
            num_rewrite_locations_considered=len(repr_location_group_ids),
            # Rewrite logprobs
            rewrite_logprobs=datapoint.get("candidate_rewrite_logprobs", None),
        )

    def initialize_minibatch(self) -> Dict[str, Any]:
        return {
            "graph_data": self.__gnn_model.initialize_minibatch(),
            # Bug Localization
            "has_bug": [],
            "correct_candidate_node_idxs": [],
            "mb_num_target_nodes": 0,
            "mb_num_rewrite_candidates": 0,
            # Repair
            "mb_num_repair_groups": 0,
            # Text rewrites
            "target_rewrites": [],
            "rewrite_to_location_group": [],
            "correct_rewrite_idxs": [],
            "text_rewrite_original_idxs": [],
            "text_rewrite_idxs": [],
            # Var Misuse
            "candidate_symbol_to_location_group": [],
            "correct_candidate_symbols": [],
            "candidate_rewrite_original_idxs": [],
            "candidate_rewrite_idxs": [],
            # Arg Swaps
            "swapped_pair_to_call_location_group": [],
            "correct_swapped_pair": [],
            "pair_rewrite_original_idx": [],
            "pair_rewrite_idxs": [],
            # Rewrite logprobs
            "rewrite_to_graph_id": [],
            "rewrite_logprobs": [],
            "no_bug_rewrite_logprobs": [],
        }

    def extend_minibatch_with(
        self, tensorized_datapoint: BaseTensorizedBugLabGnn, partial_minibatch: Dict[str, Any]
    ) -> bool:
        keep_extending = self.__gnn_model.extend_minibatch_with(
            tensorized_datapoint.graph_data, partial_minibatch["graph_data"]
        )
        if tensorized_datapoint.target_location_node_idx is None:
            partial_minibatch["has_bug"].append(False)
            partial_minibatch["correct_candidate_node_idxs"].append(0)
        else:
            partial_minibatch["has_bug"].append(True)
            partial_minibatch["correct_candidate_node_idxs"].append(
                tensorized_datapoint.target_location_node_idx + partial_minibatch["mb_num_target_nodes"]
            )
        partial_minibatch["mb_num_target_nodes"] += len(
            tensorized_datapoint.graph_data.reference_nodes["candidate_nodes"]
        )

        mb_rewrite_candidate_offset = partial_minibatch["mb_num_rewrite_candidates"]
        num_rewrite_candidates_in_datapoint = 0
        mb_num_repair_groups_offset = partial_minibatch["mb_num_repair_groups"]

        # Text rewrites
        if tensorized_datapoint.correct_rewrite_target is not None:
            partial_minibatch["correct_rewrite_idxs"].append(
                tensorized_datapoint.correct_rewrite_target + len(partial_minibatch["target_rewrites"])
            )
        partial_minibatch["target_rewrites"].extend(tensorized_datapoint.target_rewrites)
        partial_minibatch["rewrite_to_location_group"].extend(
            t + mb_num_repair_groups_offset for t in tensorized_datapoint.target_rewrite_to_location_group
        )
        partial_minibatch["text_rewrite_idxs"].extend(
            t + mb_rewrite_candidate_offset for t in tensorized_datapoint.text_rewrite_original_idx
        )
        num_rewrite_candidates_in_datapoint += len(tensorized_datapoint.text_rewrite_original_idx)

        # Var Misuse
        if tensorized_datapoint.correct_candidate_symbol_node is not None:
            partial_minibatch["correct_candidate_symbols"].append(
                tensorized_datapoint.correct_candidate_symbol_node
                + len(partial_minibatch["candidate_symbol_to_location_group"])
            )
        partial_minibatch["candidate_symbol_to_location_group"].extend(
            t + mb_num_repair_groups_offset for t in tensorized_datapoint.candidate_symbol_to_varmisused_node
        )
        partial_minibatch["candidate_rewrite_idxs"].extend(
            t + mb_rewrite_candidate_offset for t in tensorized_datapoint.candidate_rewrite_original_idx
        )
        num_rewrite_candidates_in_datapoint += len(tensorized_datapoint.candidate_rewrite_original_idx)

        # Arg Swaps
        if tensorized_datapoint.correct_swapped_pair is not None:
            partial_minibatch["correct_swapped_pair"].append(
                tensorized_datapoint.correct_swapped_pair
                + len(partial_minibatch["swapped_pair_to_call_location_group"])
            )
        partial_minibatch["swapped_pair_to_call_location_group"].extend(
            t + mb_num_repair_groups_offset for t in tensorized_datapoint.swapped_pair_to_call
        )
        partial_minibatch["pair_rewrite_idxs"].extend(
            t + mb_rewrite_candidate_offset for t in tensorized_datapoint.pair_rewrite_original_idx
        )
        num_rewrite_candidates_in_datapoint += len(tensorized_datapoint.pair_rewrite_original_idx)
        partial_minibatch["mb_num_rewrite_candidates"] += num_rewrite_candidates_in_datapoint

        partial_minibatch["mb_num_repair_groups"] += tensorized_datapoint.num_rewrite_locations_considered
        # Visualization info
        partial_minibatch["text_rewrite_original_idxs"].append(tensorized_datapoint.text_rewrite_original_idx)
        partial_minibatch["candidate_rewrite_original_idxs"].append(tensorized_datapoint.candidate_rewrite_original_idx)
        partial_minibatch["pair_rewrite_original_idx"].append(tensorized_datapoint.pair_rewrite_original_idx)

        # Rewrite logprobs
        graph_idx = len(partial_minibatch["graph_data"]["num_nodes_per_graph"]) - 1
        partial_minibatch["rewrite_to_graph_id"].extend([graph_idx] * num_rewrite_candidates_in_datapoint)
        logprobs = tensorized_datapoint.rewrite_logprobs
        if logprobs is not None:
            partial_minibatch["rewrite_logprobs"].extend(logprobs[:-1])
            partial_minibatch["no_bug_rewrite_logprobs"].append(logprobs[-1])

        return keep_extending

    def finalize_minibatch(
        self, accumulated_minibatch_data: Dict[str, Any], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        graph_data = self.__gnn_model.finalize_minibatch(accumulated_minibatch_data["graph_data"], device)

        minibatch = {
            "graph_data": graph_data,
            "correct_candidate_node_idxs": torch.tensor(
                accumulated_minibatch_data["correct_candidate_node_idxs"], dtype=torch.int64, device=device
            ),
            "has_bug": torch.tensor(accumulated_minibatch_data["has_bug"], dtype=torch.bool, device=device),
            # Text rewrites
            "target_rewrites": torch.tensor(
                accumulated_minibatch_data["target_rewrites"], dtype=torch.int64, device=device
            ),
            "rewrite_to_location_group": torch.tensor(
                accumulated_minibatch_data["rewrite_to_location_group"], dtype=torch.int64, device=device
            ),
            "correct_rewrite_idxs": torch.tensor(
                accumulated_minibatch_data["correct_rewrite_idxs"], dtype=torch.int64, device=device
            ),
            "text_rewrite_idxs": torch.tensor(
                accumulated_minibatch_data["text_rewrite_idxs"], dtype=torch.int64, device=device
            ),
            # Var Misuse
            "candidate_symbol_to_location_group": torch.tensor(
                accumulated_minibatch_data["candidate_symbol_to_location_group"], dtype=torch.int64, device=device
            ),
            "correct_candidate_symbols": torch.tensor(
                accumulated_minibatch_data["correct_candidate_symbols"], dtype=torch.int64, device=device
            ),
            "candidate_rewrite_idxs": torch.tensor(
                accumulated_minibatch_data["candidate_rewrite_idxs"], dtype=torch.int64, device=device
            ),
            # Arg swaps
            "swapped_pair_to_call_location_group": torch.tensor(
                accumulated_minibatch_data["swapped_pair_to_call_location_group"], dtype=torch.int64, device=device
            ),
            "correct_swapped_pair": torch.tensor(
                accumulated_minibatch_data["correct_swapped_pair"], dtype=torch.int64, device=device
            ),
            "pair_rewrite_idxs": torch.tensor(
                accumulated_minibatch_data["pair_rewrite_idxs"], dtype=torch.int64, device=device
            ),
            # Visualisation info
            "text_rewrite_original_idxs": accumulated_minibatch_data["text_rewrite_original_idxs"],
            "candidate_rewrite_original_idxs": accumulated_minibatch_data["candidate_rewrite_original_idxs"],
            "pair_rewrite_original_idx": accumulated_minibatch_data["pair_rewrite_original_idx"],
            # Rewrite logprobs
            "rewrite_to_graph_id": torch.tensor(
                accumulated_minibatch_data["rewrite_to_graph_id"], dtype=torch.int64, device=device
            ),
        }

        # Rewrite logprobs
        if accumulated_minibatch_data["rewrite_logprobs"]:
            rewrite_logprobs = (
                accumulated_minibatch_data["rewrite_logprobs"] + accumulated_minibatch_data["no_bug_rewrite_logprobs"]
            )
            minibatch["rewrite_logprobs"] = torch.tensor(rewrite_logprobs, dtype=torch.float32, device=device)
        return minibatch

    def predict(
        self, data: Iterator[BugLabData], trained_nn: GnnBugLabModule, device, parallelize: bool
    ) -> Iterator[Tuple[BugLabData, Dict[int, float], List[float]]]:
        trained_nn.eval()
        with torch.no_grad(), self._tensorize_all_location_rewrites():
            for mb_data, original_datapoints in self.minibatch_iterator(
                self.tensorize_dataset(data, return_input_data=True, parallelize=parallelize),
                device,
                max_minibatch_size=50,
                parallelize=parallelize,
            ):
                (
                    candidate_node_graph_idx,
                    candidate_node_log_probs,
                    gnn_output,
                    _,
                ) = trained_nn.compute_localization_logprobs(mb_data["graph_data"])

                arg_swap_logprobs, text_repair_logprobs, varmisuse_logprobs, _ = trained_nn._compute_repair_logprobs(
                    gnn_output,
                    mb_data["target_rewrites"],
                    mb_data["rewrite_to_location_group"],
                    mb_data["candidate_symbol_to_location_group"],
                    mb_data["swapped_pair_to_call_location_group"],
                )
                num_samples = gnn_output.num_graphs

                candidate_node_graph_idx = candidate_node_graph_idx.cpu().numpy()
                candidate_node_log_probs = candidate_node_log_probs.cpu().numpy()

                yield from self._iter_per_sample_results(
                    mb_data,
                    candidate_node_graph_idx,
                    candidate_node_log_probs,
                    arg_swap_logprobs,
                    num_samples,
                    original_datapoints,
                    text_repair_logprobs,
                    varmisuse_logprobs,
                )
