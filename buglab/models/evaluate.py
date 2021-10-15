#!/usr/bin/env python
"""
Usage:
    evaluate.py [options] MODEL_FILENAME TEST_DATA_PATH

Options:
    --aml                      Run this in Azure ML
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --minibatch-size=<size>    The minibatch size. [default: 300]
    --assume-buggy             Never predict NO_BUG
    --eval-only-no-bug         Evaluate only NO_BUG samples.
    --restore-path=<path>      The path to previous model file for starting from previous checkpoint.
    --limit-num-elements=<num>  Limit the number of elements to evaluate on.
    --sequential               Do not parallelize data loading. Makes debugging easier.
    --quiet                    Do not show progress bar.
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""
import math
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from docopt import docopt
from dpu_utils.utils import RichPath, run_and_debug

from buglab.models.gnn import GnnBugLabModel
from buglab.utils.msgpackutils import load_all_msgpack_l_gz


def run(arguments):
    azure_info_path = arguments.get("--azure-info", None)

    data_path = RichPath.create(arguments["TEST_DATA_PATH"], azure_info_path)

    lim = None if arguments["--limit-num-elements"] is None else int(arguments["--limit-num-elements"])
    data = load_all_msgpack_l_gz(data_path, shuffle=True, limit_num_yielded_elements=lim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(arguments["MODEL_FILENAME"])
    model, nn = GnnBugLabModel.restore_model(model_path, device)

    predictions = model.predict(data, nn, device, parallelize=not arguments["--sequential"])

    num_samples, num_location_correct = 0, 0
    num_buggy_samples, num_repaired_correct, num_repaired_given_location_correct = 0, 0, 0
    # Count warnings correct if a bug is reported, irrespectively if it's localized correctly
    num_buggy_and_raised_warning, num_non_buggy_and_no_warning = 0, 0

    localization_data_per_scout = defaultdict(lambda: np.array([0, 0], dtype=np.int32))
    repair_data_per_scout = defaultdict(lambda: np.array([0, 0], dtype=np.int32))

    bug_detection_logprobs: List[
        Tuple[float, bool, bool, bool, bool]
    ] = []  # prediction_prob, has_bug, predicted_no_bug, location_correct, rewrite_given_location_is_correct

    for datapoint, location_logprobs, rewrite_probs in predictions:
        if arguments.get("--assume-buggy", False):
            del location_logprobs[-1]
            norm = float(torch.logsumexp(torch.tensor(list(location_logprobs.values())), dim=-1))
            location_logprobs = {p: v - norm for p, v in location_logprobs.items()}

        target_fix_action_idx = datapoint["target_fix_action_idx"]
        sample_has_bug = target_fix_action_idx is not None

        if sample_has_bug and arguments.get("--eval-only-no-bug", False):
            continue
        num_samples += 1

        # Compute the predicted rewrite:
        predicted_node_idx = max(location_logprobs, key=lambda k: location_logprobs[k])
        prediction_logprob = location_logprobs[predicted_node_idx]

        predicted_rewrite_idx, predicted_rewrite_logprob = None, -math.inf
        for rewrite_idx, (rewrite_node_idx, rewrite_logprob) in enumerate(
            zip(datapoint["graph"]["reference_nodes"], rewrite_probs)
        ):
            if rewrite_node_idx == predicted_node_idx and rewrite_logprob > predicted_rewrite_logprob:
                predicted_rewrite_idx = rewrite_idx
                predicted_rewrite_logprob = rewrite_logprob

        # Compute the predicted rewrite given the correct target location:
        if not sample_has_bug:
            assert not arguments.get("--assume-buggy", False)
            ground_node_idx = -1
            target_rewrite_scout = "NoBug"
            rewrite_given_location_is_correct = None
        else:
            ground_node_idx = datapoint["graph"]["reference_nodes"][target_fix_action_idx]
            target_rewrite_scout = datapoint["candidate_rewrite_metadata"][target_fix_action_idx][0]

            predicted_rewrite_idx_given_location = None
            predicted_rewrite_logprob_given_location = -math.inf
            for rewrite_idx, (rewrite_node_idx, rewrite_logprob) in enumerate(
                zip(datapoint["graph"]["reference_nodes"], rewrite_probs)
            ):
                if rewrite_node_idx == ground_node_idx and rewrite_logprob > predicted_rewrite_logprob_given_location:
                    predicted_rewrite_idx_given_location = rewrite_idx
                    predicted_rewrite_logprob_given_location = rewrite_logprob

            rewrite_given_location_is_correct = predicted_rewrite_idx_given_location == target_fix_action_idx
            if rewrite_given_location_is_correct:
                num_repaired_given_location_correct += 1
                repair_data_per_scout[target_rewrite_scout] += [1, 1]
            else:
                repair_data_per_scout[target_rewrite_scout] += [0, 1]

            num_buggy_samples += 1

        location_is_correct = predicted_node_idx == ground_node_idx

        if location_is_correct:
            num_location_correct += 1
            localization_data_per_scout[target_rewrite_scout] += [1, 1]
        else:
            localization_data_per_scout[target_rewrite_scout] += [0, 1]

        if sample_has_bug and predicted_node_idx != -1:
            num_buggy_and_raised_warning += 1
        elif not sample_has_bug and predicted_node_idx == -1:
            num_non_buggy_and_no_warning += 1

        if location_is_correct and predicted_rewrite_idx == target_fix_action_idx:
            num_repaired_correct += 1

        bug_detection_logprobs.append(
            (
                prediction_logprob,
                sample_has_bug,
                predicted_node_idx != -1,
                location_is_correct,
                rewrite_given_location_is_correct,
            )
        )

    print("==================================")
    print(
        f"Accuracy (Localization & Repair) {num_repaired_correct/num_samples:.2%} ({num_repaired_correct}/{num_samples})"
    )
    print(
        f"Bug Detection Accuracy (no Localization or Repair) {(num_buggy_and_raised_warning + num_non_buggy_and_no_warning)/num_samples:.2%} ({num_buggy_and_raised_warning + num_non_buggy_and_no_warning}/{num_samples})"
    )
    if num_buggy_samples > 0:
        print(
            f"Bug Detection (no Localization or Repair) False Negatives: {1 - num_buggy_and_raised_warning/num_buggy_samples:.2%}"
        )
    else:
        print("Bug Detection (no Localization or Repair) False Negatives: NaN (0/0)")

    if num_samples - num_buggy_samples > 0:
        print(
            f"Bug Detection (no Localization or Repair) False Positives: {1 - num_non_buggy_and_no_warning / (num_samples - num_buggy_samples):.2%}"
        )
    else:
        print("Bug Detection (no Localization or Repair) False Positives: NaN (0/0)")

    print("==================================")
    print(f"Localization Accuracy {num_location_correct/num_samples:.2%} ({num_location_correct}/{num_samples})")
    for scout_name, (num_correct, total) in sorted(localization_data_per_scout.items(), key=lambda item: item[0]):
        print(f"\t{scout_name}: {num_correct/total:.1%}  ({num_correct}/{total})")

    print("=========================================")
    if num_buggy_samples == 0:
        print("--eval-only-no-bug is True. Repair Accuracy Given Location cannot be computed.")
    else:
        print(
            f"Repair Accuracy Given Location {num_repaired_given_location_correct/num_buggy_samples:.2%}  ({num_repaired_given_location_correct}/{num_buggy_samples})"
        )
    for scout_name, (num_correct, total) in sorted(repair_data_per_scout.items(), key=lambda item: item[0]):
        print(f"\t{scout_name}: {num_correct/total:.1%}  ({num_correct}/{total})")

    bug_detection_logprobs = sorted(bug_detection_logprobs, reverse=True)
    detection_true_warnings = np.array(
        [has_bug and correct_location for _, has_bug, _, correct_location, _ in bug_detection_logprobs]
    )

    true_warnings = np.array(
        [
            has_bug and correct_location and correct_rewrite_at_location
            for _, has_bug, _, correct_location, correct_rewrite_at_location in bug_detection_logprobs
        ]
    )

    detection_false_warnings = np.array(
        [
            predicted_is_buggy and not predicted_correct_location
            for _, has_bug, predicted_is_buggy, predicted_correct_location, _ in bug_detection_logprobs
        ]
    )
    false_warnings = np.array(
        [
            (predicted_is_buggy and not predicted_correct_location)
            or (predicted_is_buggy and not predicted_correct_rewrite_at_location)
            for _, has_bug, predicted_is_buggy, predicted_correct_location, predicted_correct_rewrite_at_location in bug_detection_logprobs
        ]
    )

    detection_true_warnings_up_to_threshold = np.cumsum(detection_true_warnings)
    detection_false_warnings_up_to_threshold = np.cumsum(detection_false_warnings)
    false_discovery_rate = detection_false_warnings_up_to_threshold / (
        detection_true_warnings_up_to_threshold + detection_false_warnings_up_to_threshold
    )

    detection_precision = detection_true_warnings_up_to_threshold / (
        detection_true_warnings_up_to_threshold + detection_false_warnings_up_to_threshold
    )
    detection_recall = detection_true_warnings_up_to_threshold / sum(
        1 for _, has_bug, _, _, _ in bug_detection_logprobs if has_bug
    )

    detection_false_no_bug_warnings = np.array(
        [
            predicted_is_buggy and not has_bug
            for _, has_bug, predicted_is_buggy, predicted_correct_location, _ in bug_detection_logprobs
        ]
    )
    no_bug_precision = 1 - np.cumsum(detection_false_no_bug_warnings) / (
        sum(1 for _, has_bug, _, _, _ in bug_detection_logprobs if has_bug) + 1e-10
    )

    threshold_x = np.linspace(0, 1, num=100)
    thresholds = np.exp(np.array([b[0] for b in bug_detection_logprobs]))
    print("x = np." + repr(threshold_x))

    print("### False Detection Rate ###")
    fdr = np.interp(threshold_x, thresholds[::-1], false_discovery_rate[::-1], right=0)
    print("fdr = np." + repr(fdr))

    print("### Detection Precision ###")
    detection_precision = np.interp(threshold_x, thresholds[::-1], detection_precision[::-1], right=0)
    print("detection_precision = np." + repr(detection_precision))

    print("### Detection Recall ###")
    detection_recall = np.interp(threshold_x, thresholds[::-1], detection_recall[::-1], right=0)
    print("detection_recall = np." + repr(detection_recall))

    print("### Detection NO_BUG Precision ###")
    no_bug_precision = np.interp(threshold_x, thresholds[::-1], no_bug_precision[::-1], right=0)
    print("no_bug_precision = np." + repr(no_bug_precision))

    true_warnings_up_to_threshold = np.cumsum(true_warnings)
    false_warnings_up_to_threshold = np.cumsum(false_warnings)

    precision = true_warnings_up_to_threshold / (true_warnings_up_to_threshold + false_warnings_up_to_threshold)
    recall = true_warnings_up_to_threshold / sum(1 for _, has_bug, _, _, _ in bug_detection_logprobs if has_bug)
    print("### Precision (Detect and Repair) ###")
    precision = np.interp(threshold_x, thresholds[::-1], precision[::-1], right=0)
    print("precision = np." + repr(precision))

    print("### Recall (Detect and Repair) ###")
    recall = np.interp(threshold_x, thresholds[::-1], recall[::-1], right=0)
    print("recall = np." + repr(recall))


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get("--debug", False))
