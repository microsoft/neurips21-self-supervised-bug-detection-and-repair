from typing import Optional

import logging
import time
from jedi.api.environment import Environment

from ..codereprs import PythonCodeRelations
from .ast import AstRelations
from .callandtypeinfo import CallAndTypeInfo
from .controlflow import ControlFlow
from .dataflow import DataFlow
from .syntactic_hyperedges import SyntacticHyperedgeRelations
from .tokens import TokenRelations

PHASES = {
    "ast": AstRelations.add_ast_relations,
    "token": TokenRelations.add_token_relations,
    "controlflow": ControlFlow.compute_controlflow_relations,
    "dataflow": DataFlow.compute_dataflow_relations,
    "call-and-type": CallAndTypeInfo.compute_call_and_type_info,
    "hyperedges": SyntacticHyperedgeRelations.add_relations,
}

LOGGER = logging.getLogger(__name__)


def compute_all_relations(rel_db: PythonCodeRelations, env: Optional[Environment] = None) -> None:
    """
    Add all the relationships in `rel_db` for the given file and environment.
    """
    timings = {}
    for phase_name, phase_fn in PHASES.items():
        start_time = time.perf_counter()
        try:
            phase_fn(rel_db, env)
        except Exception as e:
            LOGGER.exception(f"Error during {phase_name} for {rel_db.path}.", exc_info=e)
        timings[phase_name] = time.perf_counter() - start_time
    LOGGER.debug("Finished extraction for %s. Phase timing: ", rel_db.path, timings)
