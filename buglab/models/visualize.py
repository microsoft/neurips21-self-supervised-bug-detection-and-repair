#!/usr/bin/env python
"""
Usage:
    predict.py [options] MODEL_FILENAME DATA_PATH OUT_HTML

Options:
    --aml                      Run this in Azure ML
    --azure-info=<path>        Azure authentication information file (JSON). Used to load data from Azure storage.
    --minibatch-size=<size>    The minibatch size. [default: 300]
    --restore-path=<path>      The path to previous model file for starting from previous checkpoint.
    --sequential               Do not parallelize data loading. Makes debugging easier.
    --num-elements=<num>       The number of elements to visualize [default: 1000]
    --sample-pct=<num>         The percent of elements to keep from the input data. Valid for num-elements. [default: 1.]
    --only-no-bug              Print only snippets that have NO_BUG inserted.
    --only-incorrect           Print only snippets where the model has predicted something wrong.
    --order-by-confidence      Order by probability of the target action.
    --show-only-top-k=<num>    Show only the top-k. If k is zero, then show all. [default: 0]
    --quiet                    Do not show progress bar.
    -h --help                  Show this screen.
    --debug                    Enable debug routines. [default: False]
"""
import html
import io
import math
import numpy as np
import os
import pystache
import torch
from collections import defaultdict
from docopt import docopt
from dpu_utils.utils import RichPath, run_and_debug
from libcst.metadata import CodeRange
from pathlib import Path

from buglab.models.gnn import GnnBugLabModel
from buglab.utils.iteratorutils import sampled_iterator
from buglab.utils.msgpackutils import load_all_msgpack_l_gz
from buglab.utils.text import text_to_range_segments

with open(os.path.join(os.path.split(__file__)[0], "visualizationassets", "annotated_snippet.mustache")) as f:
    ANNOTATED_SNIPPET_TEMPLATE = pystache.parse(f.read())
with open(os.path.join(os.path.split(__file__)[0], "visualizationassets", "document.mustache")) as f:
    DOCUMENT_TEMPLATE = pystache.parse(f.read())


def predictions_to_html(
    predictions,
    only_show_incorrect_predictions: bool = False,
    order_results_by_confidence: bool = False,
    include_header: bool = True,
) -> str:
    renderer = pystache.Renderer()
    all_snippets = []
    for datapoint, location_logprobs, rewrite_logprobs in predictions:
        text = datapoint["graph"]["text"]
        code_range = datapoint["graph"]["code_range"]
        code_range = CodeRange((code_range[0][0], code_range[0][1]), (code_range[1][0], code_range[1][1]))
        range_to_node = {}
        range_to_actions = defaultdict(list)

        assert len(rewrite_logprobs) == len(datapoint["graph"]["reference_nodes"])

        for node_idx, rewrite, node_range, rewrite_logprob in zip(
            datapoint["graph"]["reference_nodes"],
            datapoint["candidate_rewrites"],
            datapoint["candidate_rewrite_ranges"],
            rewrite_logprobs,
        ):
            node_range = CodeRange((node_range[0][0], node_range[0][1]), (node_range[1][0], node_range[1][1]))

            range_to_node[node_range] = node_idx
            range_to_actions[node_range].append((rewrite, rewrite_logprob))

        predicted_node_idx = max(location_logprobs, key=lambda k: location_logprobs[k])

        target_fix_action_idx = datapoint["target_fix_action_idx"]
        if target_fix_action_idx is None:
            ground_node_idx = -1
            target_action = "NO_BUG"
            target_action_range = None
        else:
            ground_node_idx = datapoint["graph"]["reference_nodes"][target_fix_action_idx]
            target_action = datapoint["candidate_rewrites"][target_fix_action_idx]
            target_action_range = datapoint["candidate_rewrite_ranges"][target_fix_action_idx]
            target_action_range = CodeRange(
                (target_action_range[0][0], target_action_range[0][1]),
                (target_action_range[1][0], target_action_range[1][1]),
            )

        segments = text_to_range_segments(text, code_range, range_to_node)
        with io.StringIO() as sb:
            # Set for NO_BUG case
            is_incorrect_prediction = ground_node_idx != predicted_node_idx
            prediction_logprob = -math.inf

            for text_segment, ref_ranges in segments:
                if len(ref_ranges) == 0:
                    sb.write(html.escape(text_segment))
                    continue

                target_ranges_data = []
                for current_range in ref_ranges:
                    predicted_action, _ = max(range_to_actions[current_range], key=lambda x: x[1])

                    is_ground_truth_range = target_action_range == current_range
                    is_predicted_range = range_to_node[current_range] == predicted_node_idx
                    range_logprob = (
                        location_logprobs[range_to_node[current_range]]
                        if range_to_node[current_range] in location_logprobs
                        else math.nan
                    )
                    target_ranges_data.append(
                        {
                            "range": f"({current_range.start.line},{current_range.start.column})-({current_range.end.line},{current_range.end.column})",
                            "assigned_prob": f"{np.exp(range_logprob):.1%}",
                            "best_range_logprob": range_logprob
                            + max(rlp for _, rlp in range_to_actions[current_range]),
                            "rewrites": [
                                {
                                    "is_correct": is_ground_truth_range and rewrite == target_action,
                                    "is_predicted": is_ground_truth_range and rewrite == predicted_action,
                                    "rewrite": str(rewrite),
                                    "prob": f"{np.exp(rewrite_logprob):.1%}",
                                }
                                for rewrite, rewrite_logprob in range_to_actions[current_range]
                            ],
                            "is_ground_range": is_ground_truth_range,
                            "is_predicted_range": is_predicted_range,
                        }
                    )

                    if is_ground_truth_range and not is_predicted_range:
                        is_incorrect_prediction = True
                    elif is_ground_truth_range and is_predicted_range:
                        is_incorrect_prediction = target_action != predicted_action

                snippet_data = {
                    "text": text_segment,
                    "target_ranges": target_ranges_data,
                    "contains_ground_range": any(t["is_ground_range"] for t in target_ranges_data),
                    "contains_predicted_range": any(t["is_predicted_range"] for t in target_ranges_data),
                }
                prediction_logprob = max(prediction_logprob, max(r["best_range_logprob"] for r in target_ranges_data))
                sb.write(renderer.render(ANNOTATED_SNIPPET_TEMPLATE, snippet_data).strip())

            path = datapoint["graph"]["path"]
            if "/site-packages/" in path:
                path = path[path.find("/site-packages/") + len("/site-packages/") :]

            if only_show_incorrect_predictions and not is_incorrect_prediction:
                continue

            all_snippets.append(
                {
                    "filename": path,
                    "package": datapoint["package_name"],
                    "content": sb.getvalue(),
                    "target_action": str(target_action),
                    "no_bug_prob": f"{np.exp(location_logprobs[-1]):.1%}",
                    "is_wrong": is_incorrect_prediction,
                    "prediction_logprob": prediction_logprob,
                    "prediction_prob": f"{np.exp(prediction_logprob):.1%}",
                }
            )
    if order_results_by_confidence:
        all_snippets = sorted(all_snippets, key=lambda d: -d["prediction_logprob"])
        show_top_k = int(args["--show-only-top-k"])
        if show_top_k != 0:
            all_snippets[:show_top_k]

    return renderer.render(DOCUMENT_TEMPLATE, dict(snippets=all_snippets, include_header=include_header))


def run(arguments):
    only_show_snippets_with_no_inserted_bug = arguments["--only-no-bug"]
    only_show_incorrect_predictions = arguments["--only-incorrect"]
    order_results_by_confidence = arguments["--order-by-confidence"]

    azure_info_path = arguments.get("--azure-info", None)

    data_path = RichPath.create(arguments["DATA_PATH"], azure_info_path)
    sampling_rate = float(arguments["--sample-pct"])
    data = sampled_iterator(
        load_all_msgpack_l_gz(data_path, shuffle=sampling_rate < 1),
        num_elements=int(arguments["--num-elements"]),
        sampling_rate=sampling_rate,
    )

    if only_show_snippets_with_no_inserted_bug:
        data = (d for d in data if d["target_fix_action_idx"] is None)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = Path(arguments["MODEL_FILENAME"])
    model, nn = GnnBugLabModel.restore_model(model_path, device)

    predictions = model.predict(data, nn, device, parallelize=not arguments["--sequential"])

    rendered_html = predictions_to_html(
        predictions,
        only_show_incorrect_predictions,
        order_results_by_confidence,
    )

    with open(arguments["OUT_HTML"], "w") as f:
        f.write(rendered_html)


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args.get("--debug", False))
