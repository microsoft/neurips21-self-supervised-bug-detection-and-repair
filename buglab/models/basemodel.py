from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np
from dpu_utils.mlutils import Vocabulary
from typing_extensions import Final

from buglab.representations.data import BugLabData


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

    def _init(self):
        self._target_rewrite_ops = Vocabulary.create_vocabulary(
            self.OPERATOR_REWRITES, max_size=len(self.OPERATOR_REWRITES), count_threshold=0, add_unk=False
        )
        self._tensorize_only_at_target_location_rewrites = True

    @contextmanager
    def _tensorize_all_location_rewrites(self):
        try:
            self._tensorize_only_at_target_location_rewrites = False
            yield
        finally:
            self._tensorize_only_at_target_location_rewrites = True

    def _compute_rewrite_data(self, datapoint: BugLabData, candidate_node_idxs: List[int]):
        # TODO: This function is awful. Any way to improve it?

        target_fix_action_idx = datapoint["target_fix_action_idx"]
        if target_fix_action_idx is not None:
            target_node_idx = datapoint["graph"]["reference_nodes"][target_fix_action_idx]
        else:
            target_node_idx = None

        call_args = defaultdict(list)
        all_nodes = datapoint["graph"]["nodes"]
        for child_edge in datapoint["graph"]["edges"]["Child"]:
            if len(child_edge) == 3:
                parent_idx, child_idx, edge_type = child_edge
            else:
                parent_idx, child_idx = child_edge
                edge_type = None
            if edge_type == "args" and all_nodes[parent_idx] == "Call":
                call_args[parent_idx].append(child_idx)

        incorrect_node_id_to_text_rewrite_target: Dict[int, List[int]] = defaultdict(list)
        text_rewrite_original_idx: Dict[int, List[int]] = defaultdict(list)
        correct_text_rewrite_target: Optional[Tuple[int, int]] = None

        misused_node_ids_to_candidates_symbol_node_ids: Dict[int, List[int]] = defaultdict(list)
        varmisuse_rewrite_original_idx: Dict[int, List[int]] = defaultdict(list)
        correct_misused_node_and_candidate: Optional[Tuple[int, int]] = None

        call_node_ids_to_candidate_swapped_node_ids: Dict[int, List[int]] = defaultdict(list)
        swapped_rewrite_original_ids: Dict[int, List[int]] = defaultdict(list)
        correct_swapped_call_and_pair: Optional[Tuple[int, int]] = None

        for i, (node_idx, (rewrite_type, rewrite_data), (rewrite_scout, rewrite_metadata)) in enumerate(
            zip(
                datapoint["graph"]["reference_nodes"],
                datapoint["candidate_rewrites"],
                datapoint["candidate_rewrite_metadata"],
            )
        ):
            if self._tensorize_only_at_target_location_rewrites and node_idx != target_node_idx:
                # During training we only care about the target rewrite.
                continue
            is_target_action = target_fix_action_idx == i

            if rewrite_scout == "VariableMisuseRewriteScout":
                if is_target_action:
                    correct_misused_node_and_candidate = (
                        node_idx,
                        len(misused_node_ids_to_candidates_symbol_node_ids[node_idx]),
                    )
                misused_node_ids_to_candidates_symbol_node_ids[node_idx].append(rewrite_metadata)
                varmisuse_rewrite_original_idx[node_idx].append(i)

            elif rewrite_scout == "ArgSwapRewriteScout":
                arg_node_idxs = call_args[node_idx]
                metadata = (
                    arg_node_idxs[rewrite_data[0]],
                    arg_node_idxs[rewrite_data[1]],
                )  # The to-be swapped node idxs
                if is_target_action:
                    correct_swapped_call_and_pair = (
                        node_idx,
                        len(call_node_ids_to_candidate_swapped_node_ids[node_idx]),
                    )
                call_node_ids_to_candidate_swapped_node_ids[node_idx].append(metadata)
                swapped_rewrite_original_ids[node_idx].append(i)
            else:
                if is_target_action:
                    correct_text_rewrite_target = (
                        node_idx,
                        len(incorrect_node_id_to_text_rewrite_target[target_node_idx]),
                    )

                incorrect_node_id_to_text_rewrite_target[node_idx].append(
                    self._target_rewrite_ops.get_id_or_unk(rewrite_data)
                )
                text_rewrite_original_idx[node_idx].append(i)
        repr_location_group_ids = {location_node_idx: i for i, location_node_idx in enumerate(candidate_node_idxs)}

        def to_flat_node_selection(
            candidates: Dict[int, List],
            correct_candidate: Optional[Tuple[int, int]],
            original_rewrite_idxs: Dict[int, List[int]],
        ) -> Tuple[List[int], List[int], List[int], Optional[int], List[int]]:
            node_selection_repr_node_ids, candidate_node_ids, candidate_node_to_repr_node = [], [], []
            flat_original_rewrite_idxs: List[int] = []

            correct_idx = None
            for repr_node_id, candidate_nodes in candidates.items():
                if correct_candidate is not None and correct_candidate[0] == repr_node_id:
                    correct_idx = len(candidate_node_ids) + correct_candidate[1]
                node_selection_repr_node_ids.extend(repr_node_id for _ in candidate_nodes)
                candidate_node_ids.extend(candidate_nodes)
                group_idx = repr_location_group_ids[repr_node_id]
                flat_original_rewrite_idxs.extend(original_rewrite_idxs[repr_node_id])

                candidate_node_to_repr_node.extend((group_idx for _ in candidate_nodes))
            return (
                node_selection_repr_node_ids,
                candidate_node_ids,
                candidate_node_to_repr_node,
                correct_idx,
                flat_original_rewrite_idxs,
            )

        # Text rewrites
        (
            target_rewrite_node_ids,
            target_rewrites,
            target_rewrite_to_location_group,
            correct_rewrite_target,
            text_rewrite_original_idx,
        ) = to_flat_node_selection(
            incorrect_node_id_to_text_rewrite_target, correct_text_rewrite_target, text_rewrite_original_idx
        )
        # Single node-candidates (var misuse)
        (
            varmisused_node_ids,
            candidate_symbol_node_ids,
            candidate_symbol_to_varmisused_location,
            correct_candidate_symbol_node,
            varmisuse_rewrite_original_idx,
        ) = to_flat_node_selection(
            misused_node_ids_to_candidates_symbol_node_ids,
            correct_misused_node_and_candidate,
            varmisuse_rewrite_original_idx,
        )
        # Two node-candidates (arg swapping)
        (
            call_node_ids,
            candidate_swapped_node_ids,
            swapped_pair_to_call,
            correct_swapped_pair,
            swapped_rewrite_original_ids,
        ) = to_flat_node_selection(
            call_node_ids_to_candidate_swapped_node_ids, correct_swapped_call_and_pair, swapped_rewrite_original_ids
        )
        return (
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
        )

    def _iter_per_sample_results(
        self,
        mb_data,
        candidate_location_sample_idx,
        candidate_location_log_probs,
        arg_swap_logprobs,
        num_samples,
        original_datapoints,
        text_repair_logprobs,
        varmisuse_logprobs,
        node_mappings: List[Dict[int, int]] = None,
    ):
        localization_distribution = defaultdict(list)
        for sample_idx in range(len(candidate_location_sample_idx)):
            localization_distribution[candidate_location_sample_idx[sample_idx]].append(
                candidate_location_log_probs[sample_idx]
            )
        # The following should be an array of 1s:
        # np.sum(np.exp(np.array(list(localization_distribution.values()))), axis=-1)

        def logprobs_per_group(logprobs, rewrite_to_group):
            logprobs = logprobs.cpu().numpy()
            rewrite_to_group = rewrite_to_group.cpu().numpy()
            rewrite_probs_per_group = defaultdict(list)
            for location_group_idx, rewrite_logprob in zip(rewrite_to_group, logprobs):
                rewrite_probs_per_group[location_group_idx].append(rewrite_logprob)
            return rewrite_probs_per_group

        arg_swap_logprobs_per_group = logprobs_per_group(
            arg_swap_logprobs, mb_data["swapped_pair_to_call_location_group"]
        )
        text_repair_logprobs_per_group = logprobs_per_group(text_repair_logprobs, mb_data["rewrite_to_location_group"])
        varmisuse_logprobs_per_group = logprobs_per_group(
            varmisuse_logprobs, mb_data["candidate_symbol_to_location_group"]
        )

        next_location_group_idx = 0
        for sample_idx in range(num_samples):
            original_point: BugLabData = original_datapoints[sample_idx]
            # We rely on the fact that the candidate nodes are not reordered and appear sorted.
            ref_nodes = original_point["graph"]["reference_nodes"]
            candidate_node_idxs = np.unique(ref_nodes)
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

            text_rewrite_original_idx = mb_data["text_rewrite_original_idxs"][sample_idx]
            candidate_rewrite_original_idx = mb_data["candidate_rewrite_original_idxs"][sample_idx]
            pair_rewrite_original_idx = mb_data["pair_rewrite_original_idx"][sample_idx]

            flat_arg_swap_logprobs, flat_text_repair_logprobs, flat_var_misuse_logprob = [], [], []
            for _ in range(len(np.unique(original_point["graph"]["reference_nodes"]))):
                flat_arg_swap_logprobs.extend(arg_swap_logprobs_per_group[next_location_group_idx])
                flat_text_repair_logprobs.extend(text_repair_logprobs_per_group[next_location_group_idx])
                flat_var_misuse_logprob.extend(varmisuse_logprobs_per_group[next_location_group_idx])
                next_location_group_idx += 1

            assert len(text_rewrite_original_idx) == len(flat_text_repair_logprobs)
            assert len(candidate_rewrite_original_idx) == len(flat_var_misuse_logprob)
            assert len(pair_rewrite_original_idx) == len(flat_arg_swap_logprobs)
            rewrite_probs = [None] * len(original_point["candidate_rewrites"])

            def write_in_place(idxs, logprobs):
                for i, lprob in zip(idxs, logprobs):
                    assert rewrite_probs[i] is None
                    rewrite_probs[i] = lprob

            write_in_place(text_rewrite_original_idx, flat_text_repair_logprobs)
            write_in_place(candidate_rewrite_original_idx, flat_var_misuse_logprob)
            write_in_place(pair_rewrite_original_idx, flat_arg_swap_logprobs)

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
