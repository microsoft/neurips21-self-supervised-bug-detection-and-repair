from typing import Any, Dict, final

import argparse
import git
import json
import logging
import os
import torch
from collections import defaultdict
from dpu_utils.utils import run_and_debug
from pathlib import Path
from tempfile import TemporaryDirectory

from buglab.controllers.buggydatacreation import get_serialized_representation
from buglab.controllers.helper.concistencyfilter import BugLabWarning, detect_bug
from buglab.data.diffmining.extract import compute_relations
from buglab.models.gnn import GnnBugLabModel
from buglab.rewriting import filter_ops_in_range
from buglab.utils import detect_encoding_and_open
from buglab.utils.loggingutils import configure_logging


class PyPiBugsDataVisitor:
    def __init__(self, model, nn, device, is_hypergraphs: bool, confidence_threshold: float = 0.1):
        self._model = model
        self._nn = nn
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._is_hypergraphs = is_hypergraphs

        # Stats
        self.__num_samples = 0

        self.__num_filtered = 0
        self.__num_correctly_filtered = 0

        self.__num_false_positives = 0
        self.__num_true_positives = 0
        self.__num_false_negatives = 0
        self.__num_true_negatives = 0

    def report_metrics(self):
        print(f"Num Samples: {self.__num_samples}")
        print(f"True Positives: {self.__num_true_positives}")
        print(f"False Positives: {self.__num_false_positives}")
        print(f"True Negatives: {self.__num_true_negatives}")
        print(f"False Negatives: {self.__num_false_negatives}")
        print(
            f"Num samples filtered by counterfactual consistency: {self.__num_filtered} of which it was correct to filter {self.__num_correctly_filtered}"
        )

    @final
    def extract(self, data_path: str) -> None:
        """
        data_path: the path to the PyPiBugs metadata.
        """
        data_per_repo: Dict[str, Any] = defaultdict(list)
        with open(data_path) as f:
            for line in f:
                line = json.loads(line)
                data_per_repo[line["repo"]].append(line)

        for i, (repo_url, data) in enumerate(data_per_repo.items()):
            if i % 20 == 19:
                self.report_metrics()

            with TemporaryDirectory() as tmp_dir:
                try:
                    logging.info("Cloning %s", repo_url)
                    repo: git.Repo = git.Repo.clone_from(repo_url, tmp_dir)
                except Exception as e:
                    logging.exception("Error in cloning ", exc_info=e)
                    continue
                logging.info("Traversing commits in %s", repo_url)
                # Clone repo

                for bug_data in data:
                    try:
                        commit = repo.commit(bug_data["hash"])
                        parents = commit.parents
                        assert len(parents) == 1, "All PyPi bugs should have a single parent"

                        # Checkout
                        parent_commit: git.Commit = parents[0]
                        repo.git.reset("--hard")
                        repo.git.checkout(parent_commit)

                        # Invoke before
                        target_file_path = os.path.join(tmp_dir, bug_data["old_path"])
                        buggy_data = self.visit_buggy_code(tmp_dir, target_file_path, bug_data, commit)

                        # Invoke after
                        repo.git.reset("--hard")
                        repo.git.checkout(commit)
                        for diff in commit.diff(parent_commit):
                            if diff.a_path == bug_data["old_path"]:
                                target_file_path = os.path.join(tmp_dir, diff.b_path)
                                break
                        else:
                            logging.error("Should never reach here. Could not find path of input file")

                        self.visit_fixed_code(tmp_dir, target_file_path, bug_data, commit, buggy_data)
                    except Exception as e:
                        logging.exception("Error", exc_info=e)

    def compute_warnings(self, bug_fixing_commit, repo_path, target_file_path, target_rewrite):
        with detect_encoding_and_open(target_file_path) as f:
            code_text = f.read()
        available_ops, available_ops_metadata, function_finder, rel_db = compute_relations(code_text, target_file_path)
        for function, fn_pos in function_finder.all_function_nodes:
            relevant_ops, relevant_op_metadata = filter_ops_in_range(available_ops, available_ops_metadata, fn_pos)
            for op in relevant_ops:
                if str(op) == target_rewrite:
                    target_fix_op = op
                    break
            else:
                continue
            break
        else:
            raise Exception("Could not find relevant function.")
        fn_data = get_serialized_representation(
            rel_db, fn_pos, relevant_ops, relevant_op_metadata, target_fix_op, repo_path, str(bug_fixing_commit)
        )
        analysis_result, op = detect_bug(
            self._model,
            self._nn,
            self._device,
            fn_data,
            relevant_ops,
            target_file_path,
            venv_location=None,
            confidence_threshold=self._confidence_threshold,
            as_hypergraphs=self._is_hypergraphs,
        )
        _, revert_op = target_fix_op.rewrite(code_text)
        return analysis_result, op, revert_op

    def visit_buggy_code(
        self, repo_path: str, target_file_path: str, bug_metadata, bug_fixing_commit: git.Commit
    ) -> None:
        """
        Invoked with the repository checked out at the state _before_ the bug-fixing commit.
        """
        target_rewrite = bug_metadata["rewrite"]
        analysis_result, op, revert_op = self.compute_warnings(
            bug_fixing_commit, repo_path, target_file_path, target_rewrite
        )

        self.__num_samples += 1

        if analysis_result == BugLabWarning.CONSISTENT_WARNING:
            if str(op) == target_rewrite:
                self.__num_true_positives += 1
                logging.info("True Positive")
            elif op is None:
                self.__num_false_negatives += 1
                logging.info(f"False Negative. No bug detected instead of `{target_rewrite}`.")
            else:
                self.__num_false_positives += 1
                logging.info(f"False Positive. Incorrect bug detected `{op}` instead of `{target_rewrite}`.")
        else:
            if analysis_result == BugLabWarning.INCONSISTENT_WARNING:
                self.__num_filtered += 1
                if str(op) != target_rewrite:
                    self.__num_correctly_filtered += 1
                else:
                    print(f"Consistency filter incorrectly removed {op}")
            self.__num_false_negatives += 1
            logging.info(f"False Negative. No bug detected instead of `{target_rewrite}`.")

        return revert_op

    def visit_fixed_code(
        self, repo_path: str, target_file_path, bug_metadata, bug_fixing_commit: git.Commit, bug_introducing_op
    ) -> None:
        """
        Invoked with the repository checked out at the state _after_ the bug-fixing commit.
        """
        analysis_result, op, _ = self.compute_warnings(
            bug_fixing_commit, repo_path, target_file_path, str(bug_introducing_op)
        )

        self.__num_samples += 1

        if op is None or analysis_result != BugLabWarning.CONSISTENT_WARNING:
            self.__num_true_negatives += 1
            logging.info("True Negative.")
        else:
            self.__num_false_positives += 1
            logging.info(f"False Positive. No bug should have been detected. Instead `{op}` was detected,")

        if analysis_result == BugLabWarning.INCONSISTENT_WARNING:
            self.__num_filtered += 1
            self.__num_correctly_filtered += 1


def run(arguments):
    confidence_threshold = arguments.confidence_level
    assert 0 <= confidence_threshold < 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = Path(arguments.model_path)
    model, nn = GnnBugLabModel.restore_model(model_path, device)

    extractor = PyPiBugsDataVisitor(model, nn, device, arguments.hypergraph, confidence_threshold)
    extractor.extract(arguments.pypi_bugs_dir)

    extractor.report_metrics()


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description="Filter warnings using unit tests.")

    parser.add_argument("pypi_bugs_dir", type=str, help="the directory where the package is located.")

    parser.add_argument("model_path", type=str, help="the path to the BugLab model.")
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.1,
        help="The confidence threshold for deciding if a warning is to be tested.",
    )

    parser.add_argument(
        "--venv-location",
        type=str,
        help="The virtual environment location to run tests in.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug on exception.",
    )

    parser.add_argument(
        "--hypergraph",
        action="store_true",
        help="Model is hypergraph model",
    )

    args = parser.parse_args()
    run_and_debug(lambda: run(args), args.debug)
