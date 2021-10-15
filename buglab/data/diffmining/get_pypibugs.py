import json
import logging
import os
from collections import defaultdict
from tempfile import TemporaryDirectory
from typing import Any, Dict, final

import git


class PyPiBugsDataVisitor:
    """
    To extract the PyPiBugs dataset, you can use this class as the main scaffold.

    The code automatically, reads in the PyPiBugs dataset definition, clones
    the repos and checkouts out the appropriate commits.

    The `visit_buggy_code` and `visit_fixed_code` need to be implemented:
    * `visit_buggy_code` is called on the version of the code before fixing the bug.
        The full repository and is accessible at the `repo_path` argument.
    * `visit_fixed_code` is called immediately after `visit_buggy_code` and the
        repository is at the version after the bug is fixed.
    """

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

        for repo_url, data in data_per_repo.items():
            with TemporaryDirectory() as tmp_dir:
                logging.info("Cloning %s", repo_url)
                repo: git.Repo = git.Repo.clone_from(repo_url, tmp_dir)
                logging.info("Traversing commits in %s", repo_url)
                # Clone repo

                for bug_data in data:
                    commit = repo.commit(bug_data["sha"])
                    parents = commit.parents
                    assert len(parents) == 1, "All PyPi bugs should have a single parent"

                    # Checkout
                    parent_commit: git.Commit = parents[0]
                    repo.git.checkout(parent_commit)

                    # Invoke before
                    target_file_path = os.path.join(tmp_dir, bug_data["path"])
                    self.visit_buggy_code(tmp_dir, target_file_path, bug_data, commit)

                    # Invoke after
                    repo.git.checkout(commit)
                    for diff in commit.diff(parent_commit):
                        if diff.a_path == bug_data["path"]:
                            target_file_path = os.path.join(tmp_dir, diff.b_path)
                            break
                    else:
                        logging.error("Should never reach here. Could not find path of input file")

                    self.visit_fixed_code(tmp_dir, target_file_path, bug_data, commit)

    def visit_buggy_code(
        self, repo_path: str, target_file_path: str, bug_metadata, bug_fixing_commit: git.Commit
    ) -> None:
        """
        Invoked with the repository checked out at the state _before_ the bug-fixing commit.
        """
        ...  # TODO: Implement your data extraction code here.

    def visit_fixed_code(self, repo_path: str, target_file_path, bug_metadata, bug_fixing_commit: git.Commit) -> None:
        """
        Invoked with the repository checked out at the state _after_ the bug-fixing commit.
        """
        ...  # TODO: Implement you data extraction code here


if __name__ == "__main__":
    import sys

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    extractor = PyPiBugsDataVisitor()
    extractor.extract(sys.argv[1])
