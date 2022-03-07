import argparse
import libcst as cst
import logging
import os
import sys
from dpu_utils.utils import run_and_debug
from pathlib import Path

from buglab.controllers.buggydatacreation import extract_for_package
from buglab.data.deduplication import DuplicationClient
from buglab.rewriting import AssignRewriteScout, BinaryOperatorRewriteScout
from buglab.utils.loggingutils import configure_logging
from buglab.utils.msgpackutils import save_msgpack_l_gz

LOGGER = logging.getLogger(__name__)


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description="Orchestrator to extract (static graphs) across multiple packages.")
    parser.add_argument("package", type=str, help="The target package")
    parser.add_argument("--target-dir", type=str, default="/data/targetDir", help="The path for the extracted data.")
    parser.add_argument("--debug", action="store_true", help="Enter debugging mode when an exception is thrown.")
    parser.add_argument(
        "--deduplication-server",
        type=str,
        default="tcp://localhost:5555",
        help="The zmq address to the deduplication server.",
    )
    parser.add_argument(
        "--bug-selector-server",
        type=str,
        default="tcp://localhost:5556",
        help="The zmq address to the bug selector server.",
    )
    parser.add_argument(
        "--num-semantics-preserving-transforms",
        type=str,
        default=0,
        help="Number of semantics-preserving transformation on the input files.",
    )
    args = parser.parse_args()

    # Disable "easy" rewrite ops
    AssignRewriteScout.DISABLED_OPS = frozenset(
        {
            cst.MultiplyAssign,
            cst.PowerAssign,
            cst.DivideAssign,
            cst.FloorDivideAssign,
            cst.ModuloAssign,
            cst.MatrixMultiplyAssign,
            cst.LeftShiftAssign,
            cst.RightShiftAssign,
            cst.BitOrAssign,
            cst.BitAndAssign,
            cst.BitXorAssign,
        }
    )

    BinaryOperatorRewriteScout.DISABLED_BINOPS = frozenset(
        {
            cst.Power,
            cst.FloorDivide,
            cst.Modulo,
            cst.MatrixMultiply,
            cst.LeftShift,
            cst.RightShift,
            cst.BitOr,
            cst.BitAnd,
            cst.BitXor,
        }
    )

    os.makedirs(args.target_dir, exist_ok=True)

    target_path = Path(args.target_dir) / f"graphs-{args.package}.msgpack.l.gz"

    def flatten():
        extacted_data_iter = extract_for_package(
            package_name=args.package,
            bug_selector_server_address=args.bug_selector_server,
            deduplication_client=DuplicationClient(args.deduplication_server),
            num_semantics_preserving_transformations_per_file=args.num_semantics_preserving_transforms,
        )
        for extacted_function_code in extacted_data_iter:
            for extracted_e, _ in extacted_function_code["rewrites"].values():
                yield extracted_e

    run_and_debug(
        lambda: save_msgpack_l_gz(
            flatten(),
            target_path,
        ),
        args.debug,
    )
    sys.exit(0)
