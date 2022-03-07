from typing import Any, Callable, DefaultDict, Dict, Iterator, List, Literal, NamedTuple, Optional, Tuple, Union
from typing_extensions import Final

import numpy as np
import torch
from collections import defaultdict
from dpu_utils.mlutils import Vocabulary

from buglab.models.rewrite_chooser_module import RewriteChooserInformation, RewriteChooserModule, RewriteLogprobs
from buglab.representations.data import BugLabData, GraphData, HypergraphData


class RewriteModuleSelectionInformation(NamedTuple):
    # The nodes at which we can rewrite:
    node_idxs: List[int]
    # The potential replacements (aligned with `node_idxs`); can be IDs into vocab or other nodes
    replacement_ids: List[int]
    # Grouping index; same for all rewrites at same location (aligned with `node_idxs`):
    loc_groups: List[int]
    # The index of the correct choice in a given location group (if any such exists)
    correct_choice_idx: Optional[int]
    # The original index of the rewrite (in the set of rewrites for all modules)
    original_idxs: List[int]


class AbstractBugLabModel:
    OPERATOR_REWRITES: Final = frozenset(
        {
            "+",
            "-",
            "*",
            "/",
            "**",
            "//",
            "%",
            "@",
            "<<",
            ">>",
            "|",
            "&",
            "^",
            "+=",
            "-=",
            "*=",
            "/=",
            "**=",
            "//=",
            "%=",
            "@=",
            "<<=",
            ">>=",
            "|=",
            "&=",
            "^=",
            "=",
            "<",
            "<=",
            ">",
            ">=",
            "==",
            "!=",
            " in ",
            " not in ",
            " is ",
            " is not ",
            "0",
            "1",
            "2",
            "-1",
            "-2",
            "and",
            "or",
            "not ",
            "",
            "True",
            "False",
        }
    )  # TODO: This is language specific

    def __init__(
        self,
        *,
        generator_loss_type: str = "classify-max-loss",
        buggy_samples_weight_schedule: Callable[[int], float] = lambda _: 1.0,
        repair_weight_schedule: Callable[[int], float] = lambda _: 1.0,
        localization_module_type: Literal[
            "CandidateQuery", "RewriteQuery", "CandidateAndRewriteQuery"
        ] = "CandidateAndRewriteQuery",
    ):
        super().__init__()
        self._init()
        self._generator_loss_type = generator_loss_type
        self._buggy_samples_weight_schedule = buggy_samples_weight_schedule
        self._repair_weight_schedule = repair_weight_schedule
        self._localization_module_type = localization_module_type

    def _init(self):
        self._target_rewrite_ops = Vocabulary.create_vocabulary(
            self.OPERATOR_REWRITES, max_size=len(self.OPERATOR_REWRITES), count_threshold=0, add_unk=False
        )

    def _compute_rewrite_data(self, datapoint: BugLabData, candidate_node_idxs: List[int]):
        # TODO: This function is awful. Any way to improve it?

        target_fix_action_idx = datapoint["target_fix_action_idx"]
        if target_fix_action_idx is not None:
            target_node_idx = datapoint["graph"]["reference_nodes"][target_fix_action_idx]
        else:
            target_node_idx = None

        call_args = defaultdict(list)
        all_nodes = datapoint["graph"]["nodes"]
        for child_edge in datapoint["graph"]["edges"].get("Child", []):
            if len(child_edge) == 3:
                parent_idx, child_idx, edge_type = child_edge
            else:
                parent_idx, child_idx = child_edge
                edge_type = None
            if edge_type == "args" and all_nodes[parent_idx] == "Call":
                call_args[parent_idx].append(child_idx)

        if "Child" not in datapoint["graph"]["edges"]:
            # We are looking at hyperedges
            for hyperedge in datapoint["graph"]["hyperedges"]:
                # {'name': '$AstChild', 'docstring': None, 'args': {'func': [52], 'args': [51], '$rval': [49]}}
                if hyperedge["name"] == "Child" and hyperedge.get("argtype", None) is not None:
                    parent_idx = hyperedge["args"]["from"][0]
                    child_idx = hyperedge["args"]["to"][0]
                    edge_type = hyperedge["argtype"]
                    if edge_type == "args" and all_nodes[parent_idx] == "Call":
                        call_args[parent_idx].append(child_idx)
                if (
                    hyperedge["name"] != "$AstChild"
                    or "func" not in hyperedge["args"]
                    or "args" not in hyperedge["args"]
                ):
                    continue

                arg_idx = 0
                while True:
                    argname = "args" + ("" if arg_idx == 0 else str(arg_idx))
                    if argname not in hyperedge["args"]:
                        break
                    call_args[hyperedge["args"]["$rval"][0]].extend(hyperedge["args"][argname])
                    arg_idx += 1

        text_rewrite_node_to_rewrite_id: Dict[int, List[int]] = defaultdict(list)
        text_rewrite_original_idxs: Dict[int, List[int]] = defaultdict(list)
        text_rewrite_correct_node_and_id: Optional[Tuple[int, int]] = None

        varswap_node_ids_to_candidate_node_ids: Dict[int, List[int]] = defaultdict(list)
        varswap_original_idx: Dict[int, List[int]] = defaultdict(list)
        varswap_correct_node_and_candidate: Optional[Tuple[int, int]] = None

        argswap_node_idxs_to_arg_replacement_node_idxs: Dict[int, List[int]] = defaultdict(list)
        argswap_original_idx: Dict[int, List[int]] = defaultdict(list)
        argswap_correct_node_and_candidate: Optional[Tuple[int, int]] = None

        for i, (node_idx, (rewrite_type, rewrite_data), (rewrite_scout, rewrite_metadata)) in enumerate(
            zip(
                datapoint["graph"]["reference_nodes"],
                datapoint["candidate_rewrites"],
                datapoint["candidate_rewrite_metadata"],
            )
        ):
            is_target_action = target_fix_action_idx == i

            if rewrite_scout == "VariableMisuseRewriteScout":
                if is_target_action:
                    varswap_correct_node_and_candidate = (
                        node_idx,
                        len(varswap_node_ids_to_candidate_node_ids[node_idx]),
                    )
                varswap_node_ids_to_candidate_node_ids[node_idx].append(rewrite_metadata)
                varswap_original_idx[node_idx].append(i)
            elif rewrite_scout == "ArgSwapRewriteScout":
                arg_node_idxs = call_args[node_idx]
                metadata = (
                    arg_node_idxs[rewrite_data[0]],
                    arg_node_idxs[rewrite_data[1]],
                )  # The to-be swapped node idxs
                if is_target_action:
                    argswap_correct_node_and_candidate = (
                        node_idx,
                        len(argswap_node_idxs_to_arg_replacement_node_idxs[node_idx]),
                    )
                argswap_node_idxs_to_arg_replacement_node_idxs[node_idx].append(metadata)
                argswap_original_idx[node_idx].append(i)
            else:
                if is_target_action:
                    text_rewrite_correct_node_and_id = (
                        node_idx,
                        len(text_rewrite_node_to_rewrite_id[target_node_idx]),
                    )

                text_rewrite_node_to_rewrite_id[node_idx].append(self._target_rewrite_ops.get_id_or_unk(rewrite_data))
                text_rewrite_original_idxs[node_idx].append(i)
        repr_location_group_ids = {location_node_idx: i for i, location_node_idx in enumerate(candidate_node_idxs)}

        def to_flat_node_selection(
            candidates: Dict[int, List],
            correct_candidate: Optional[Tuple[int, int]],
            original_rewrite_idxs: Dict[int, List[int]],
        ) -> RewriteModuleSelectionInformation:
            node_selection_repr_node_ids, candidate_node_ids, candidate_node_to_group_id = [], [], []
            flat_original_rewrite_idxs: List[int] = []

            correct_idx = None
            for repr_node_id, candidate_nodes in candidates.items():
                if correct_candidate is not None and correct_candidate[0] == repr_node_id:
                    correct_idx = len(candidate_node_ids) + correct_candidate[1]
                node_selection_repr_node_ids.extend(repr_node_id for _ in candidate_nodes)
                candidate_node_ids.extend(candidate_nodes)
                group_idx = repr_location_group_ids[repr_node_id]
                flat_original_rewrite_idxs.extend(original_rewrite_idxs[repr_node_id])

                candidate_node_to_group_id.extend((group_idx for _ in candidate_nodes))
            return RewriteModuleSelectionInformation(
                node_idxs=node_selection_repr_node_ids,
                replacement_ids=candidate_node_ids,
                loc_groups=candidate_node_to_group_id,
                correct_choice_idx=correct_idx,
                original_idxs=flat_original_rewrite_idxs,
            )

        num_rewrite_locations = len(repr_location_group_ids)
        return (
            to_flat_node_selection(
                text_rewrite_node_to_rewrite_id, text_rewrite_correct_node_and_id, text_rewrite_original_idxs
            ),
            to_flat_node_selection(
                varswap_node_ids_to_candidate_node_ids, varswap_correct_node_and_candidate, varswap_original_idx
            ),
            to_flat_node_selection(
                argswap_node_idxs_to_arg_replacement_node_idxs, argswap_correct_node_and_candidate, argswap_original_idx
            ),
            num_rewrite_locations,
        )

    def _iter_per_sample_results(
        self,
        mb_data: Dict[str, Any],
        rewrite_info: RewriteChooserInformation,
        rewrite_logprobs: RewriteLogprobs,
        original_datapoints: List[Optional[BugLabData]],
    ):
        localization_distribution: DefaultDict[int, float] = defaultdict(list)
        for loc_idx, loc_logprob in enumerate(rewrite_logprobs.localization_logprobs):
            # The last num_samples entries in localization_logprobs are for the virtual "NO_BUG" locations
            # which aren't in the loc_to_sample map:
            if loc_idx < len(rewrite_info.rewritable_loc_to_sample_id):
                sample_idx = rewrite_info.rewritable_loc_to_sample_id[loc_idx].item()
            else:
                sample_idx = loc_idx - len(rewrite_info.rewritable_loc_to_sample_id)
            localization_distribution[sample_idx].append(loc_logprob.item())
        # The following should be an array of 1s:
        # np.sum(np.exp(np.array(list(localization_distribution.values()))), axis=-1)

        def logprobs_per_group(logprobs, rewrite_to_group):
            logprobs = logprobs.cpu().numpy()
            rewrite_to_group = rewrite_to_group.cpu().numpy()
            rewrite_probs_per_group = defaultdict(list)
            for location_group_idx, rewrite_logprob in zip(rewrite_to_group, logprobs):
                rewrite_probs_per_group[location_group_idx].append(rewrite_logprob)
            return rewrite_probs_per_group

        text_repair_logprobs_per_group = logprobs_per_group(
            rewrite_logprobs.text_rewrite_logprobs, mb_data["text_rewrite_to_loc_group"]
        )
        varmisuse_logprobs_per_group = logprobs_per_group(
            rewrite_logprobs.varswap_logprobs, mb_data["varswap_to_loc_group"]
        )
        arg_swap_logprobs_per_group = logprobs_per_group(
            rewrite_logprobs.argswap_logprobs, mb_data["argswap_to_loc_group"]
        )

        next_location_group_idx = 0
        for sample_idx in range(rewrite_info.num_samples):
            original_point: Optional[BugLabData] = original_datapoints[sample_idx]
            if original_point is None:
                raise ValueError("Expected to have access to original datapoint")
            # We rely on the fact that the candidate nodes are not reordered and appear sorted.
            ref_nodes = original_point["graph"]["reference_nodes"]
            candidate_node_idxs = np.unique(ref_nodes)
            node_mappings = mb_data.get("node_mappings")
            if node_mappings is not None:
                # ref_nodes = [node_mappings[sample_idx][k] for k in ref_nodes] <- unused, but keep in mind.
                candidate_node_idxs = np.array([node_mappings[sample_idx][k] for k in candidate_node_idxs])

            # Note: the above is _not_ the same as mapping the ref_nodes and then calling
            # candidate_node_idxs=np.unique(ref_nodes)
            # There may well be duplicate candidate_node_idxs here thanks to the non-unique mapping
            # of nodes from the graph to the sequence. This non-uniqueness needs to be taken into
            # account in the zip call below.

            assert len(localization_distribution[sample_idx]) == len(candidate_node_idxs) + 1

            location_logprobs = {
                node_idx: lprob for node_idx, lprob in zip(candidate_node_idxs, localization_distribution[sample_idx])
            }
            location_logprobs[-1] = localization_distribution[sample_idx][-1]

            text_rewrite_original_idxs = mb_data["text_rewrite_original_idxs"][sample_idx]
            candidate_rewrite_original_idx = mb_data["varswap_original_idxs"][sample_idx]
            argswap_original_idxs = mb_data["argswap_original_idxs"][sample_idx]

            flat_arg_swap_logprobs, flat_text_repair_logprobs, flat_var_misuse_logprob = [], [], []
            for _ in range(len(np.unique(original_point["graph"]["reference_nodes"]))):
                flat_arg_swap_logprobs.extend(arg_swap_logprobs_per_group[next_location_group_idx])
                flat_text_repair_logprobs.extend(text_repair_logprobs_per_group[next_location_group_idx])
                flat_var_misuse_logprob.extend(varmisuse_logprobs_per_group[next_location_group_idx])
                next_location_group_idx += 1

            assert len(text_rewrite_original_idxs) == len(flat_text_repair_logprobs)
            assert len(candidate_rewrite_original_idx) == len(flat_var_misuse_logprob)
            assert len(argswap_original_idxs) == len(flat_arg_swap_logprobs)
            rewrite_probs = [None] * len(original_point["candidate_rewrites"])

            def write_in_place(idxs, logprobs):
                for i, lprob in zip(idxs, logprobs):
                    assert rewrite_probs[i] is None
                    rewrite_probs[i] = lprob

            write_in_place(text_rewrite_original_idxs, flat_text_repair_logprobs)
            write_in_place(candidate_rewrite_original_idx, flat_var_misuse_logprob)
            write_in_place(argswap_original_idxs, flat_arg_swap_logprobs)

            assert None not in rewrite_probs
            if node_mappings is not None:
                reverse_map = defaultdict(list)
                for old, new in node_mappings[sample_idx].items():
                    if old not in original_point["graph"]["reference_nodes"]:
                        continue
                    reverse_map[new].append(old)

                new_location_logprobs = {}
                for n, p in location_logprobs.items():
                    if n >= 0:
                        nodes = reverse_map[n]
                    else:
                        nodes = [n]
                    for node in nodes:
                        new_location_logprobs[node] = p
                location_logprobs = new_location_logprobs

            # At this point
            # np.sum(np.exp(list(location_logprobs.values())))
            # should be equal to 1 (up to floating point errors).

            yield original_point, location_logprobs, rewrite_probs

    def predict(
        self, data: Iterator[BugLabData], trained_nn: RewriteChooserModule, device, parallelize: bool
    ) -> Iterator[Tuple[BugLabData, Dict[int, float], List[float]]]:
        trained_nn.eval()
        with torch.no_grad():
            for mb_data, original_datapoints in self.minibatch_iterator(
                self.tensorize_dataset(data, return_input_data=True, parallelize=parallelize),
                device,
                max_minibatch_size=50,
                parallelize=parallelize,
            ):
                rc_info = trained_nn.compute_rewrite_chooser_information(**mb_data)
                rewrite_logprobs = trained_nn.compute_rewrite_logprobs(rc_info)
                yield from self._iter_per_sample_results(
                    mb_data,
                    rewrite_info=rc_info,
                    rewrite_logprobs=rewrite_logprobs,
                    original_datapoints=original_datapoints,
                )
