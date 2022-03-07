import argparse
import logging
import msgpack
import time
import zmq
from dpu_utils.utils import RichPath
from libcst.metadata import CodeRange
from threading import Thread
from tqdm import tqdm

from buglab.controllers.datageneratingpipeline_coordinator import data_pipeline_proxy
from buglab.representations.data import BugLabData
from buglab.utils.cstutils import relative_range
from buglab.utils.loggingutils import configure_logging
from buglab.utils.msgpackutils import load_all_msgpack_l_gz
from buglab.utils.text import get_text_in_range

LOGGER = logging.getLogger(__name__)


def get_data_from_folder(
    path: str,
):
    # We assume that the files have been created in order: First a NO_BUG, then the various buggy versions
    # (as in the packageextracttodisk.py).
    current_fn = None
    fn_samples = {}
    for sample in tqdm(load_all_msgpack_l_gz(RichPath.create(path))):
        sample: BugLabData

        fn_name = (
            f'{sample["graph"]["path"]}:{sample["graph"]["code_range"][0][0]}-{sample["graph"]["code_range"][0][1]}'
        )

        if fn_name != current_fn and len(fn_samples) > 0:
            yield {
                "original": fn_samples["NO_BUG"],
                "rewrites": {str(p): (s, 1 / len(fn_samples)) for p, s in fn_samples.items()},
            }

        if fn_name != current_fn:
            # Reset
            current_fn = fn_name
            fn_samples = {}

        if sample["target_fix_action_idx"] is None:
            fn_samples["NO_BUG"] = sample
        else:
            # Find the target rewrite on the original since we forgot to keep this information.
            if "NO_BUG" not in fn_samples:
                # Might happen towards the beginning of files.
                LOGGER.warning("Found sample without the original. Ignoring...")
                continue

            inverse_rewrite = sample["candidate_rewrites"][sample["target_fix_action_idx"]]
            inverse_rewrite_range = sample["candidate_rewrite_ranges"][sample["target_fix_action_idx"]]

            if inverse_rewrite[0] == "ReplaceText":
                inverse_op_range = CodeRange(tuple(inverse_rewrite_range[0]), tuple(inverse_rewrite_range[1]))
                snippet_range = sample["graph"]["code_range"]
                relative_r = relative_range(
                    CodeRange(tuple(snippet_range[0]), tuple(snippet_range[1])), inverse_op_range
                )
                buggy_text = get_text_in_range(sample["graph"]["text"], relative_r)

                for idx, ((rewrite_op, rewrite_args), rewrite_range) in enumerate(
                    zip(fn_samples["NO_BUG"]["candidate_rewrites"], fn_samples["NO_BUG"]["candidate_rewrite_ranges"])
                ):
                    if rewrite_op != "ReplaceText":
                        continue
                    if rewrite_range[0] != inverse_rewrite_range[0]:  # Rewrites should start at the same position.
                        continue
                    if rewrite_args == buggy_text:
                        fn_samples[idx] = sample
                        break
                else:
                    raise Exception("Inverse not found.")

            elif inverse_rewrite[0] == "ArgSwap":
                for idx, ((rewrite_op, rewrite_args), rewrite_range) in enumerate(
                    zip(fn_samples["NO_BUG"]["candidate_rewrites"], fn_samples["NO_BUG"]["candidate_rewrite_ranges"])
                ):
                    if rewrite_op != "ArgSwap":
                        continue
                    if rewrite_range != inverse_rewrite_range:
                        continue
                    if rewrite_args == inverse_rewrite[1]:
                        fn_samples[idx] = sample
                        break
                else:
                    raise Exception("Inverse not found.")
            else:
                raise Exception("Unknown rewrite type.")


def publish_data_from_folder(
    path: str,
    data_publishing_proxy_address: str,
    throttle_num_sent_per_sec: float = 10,
):
    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)
    socket.connect(data_publishing_proxy_address)

    for data in get_data_from_folder(path):
        time.sleep(1 / throttle_num_sent_per_sec)  # Throttle
        socket.send(msgpack.dumps(data))


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(
        description="A publisher that simply publishes files from a folder, "
        "assigning random scores to each bug. Used for debugging."
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="the path to a txt file containing the names of the packages to be considered",
    )

    parser.add_argument(
        "--throttle-elements-sent-per-sec",
        type=float,
        default=10,
        help="The max number of elements to be sent per-second.",
    )

    args = parser.parse_args()
    LOGGER.info("Run args: %s", args)

    proxy_thread = Thread(target=data_pipeline_proxy, daemon=True)
    proxy_thread.start()
    while True:
        publish_data_from_folder(args.data_path, "tcp://localhost:5557", args.throttle_elements_sent_per_sec)
