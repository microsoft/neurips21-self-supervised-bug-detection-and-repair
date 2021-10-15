import argparse
import os
from typing import Any, List, Optional, Tuple, Type

import graphviz as gv
import jedi
import libcst as cst
import streamlit as st

st.set_page_config(layout="wide")
from glob import glob
from pathlib import Path

import torch

from buglab.controllers.buggydatacreation import get_serialized_representation
from buglab.models.gnn import GnnBugLabModel
from buglab.models.visualize import predictions_to_html
from buglab.representations.coderelations import compute_all_relations
from buglab.representations.codereprs import PythonCodeRelations
from buglab.rewriting import ALL_REWRITE_SCOUTS, AbstractRewriteOp, ICodeRewriteScout, filter_ops_in_range
from buglab.utils.cstutils import AllFunctionFinder


@st.cache(allow_output_mutation=True)
def load_model(model_path: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = Path(model_path)
    model, nn = GnnBugLabModel.restore_model(model_path, device)
    return device, model, nn


EDGE_COLORS = {
    "LastMayWrite": "red",
    "NextMayUse": "blue",
    "OccurrenceOf": "green",
    "ComputedFrom": "brown",
    "NextToken": "hotpink",
    "Sibling": "lightgray",
}


def get_data_iterator_from(code_text: str, venv_location: Optional[str] = None):
    jedi_env = jedi.create_environment(venv_location) if venv_location else None

    rel_db = PythonCodeRelations(code_text, Path("unk_path"))
    compute_all_relations(rel_db, jedi_env)

    available_ops: List[AbstractRewriteOp] = []
    available_ops_metadata: List[Tuple[Type[ICodeRewriteScout], cst.CSTNode, Any]] = []
    rel_db.ast_with_metadata_wrapper.visit_batched(
        [ScoutClass(available_ops, available_ops_metadata) for ScoutClass in ALL_REWRITE_SCOUTS]
    )

    function_finder = AllFunctionFinder()
    rel_db.ast_with_metadata_wrapper.visit(function_finder)
    for fn_node, fn_range in function_finder.all_function_nodes:
        relevant_ops, relevant_op_metadata = filter_ops_in_range(available_ops, available_ops_metadata, fn_range)
        yield get_serialized_representation(
            rel_db,
            fn_range,
            relevant_ops,
            relevant_op_metadata,
            target_fix_op=None,
            package_name="unknown package",
            package_version="unknown",
        )


parser = argparse.ArgumentParser(description="An explorer tool.")

parser.add_argument("models_dir", type=str, help="the path to the trained model")

parser.add_argument(
    "--venv-path",
    type=str,
    help="The path to the appropriate virtual environment.",
)

args = parser.parse_args()

model_path = st.selectbox("Model", options=glob(os.path.join(args.models_dir, "**", "*.pkl.gz"), recursive=True))

device, model, nn = load_model(model_path)


@st.cache
def predict(code_text):
    predictions = model.predict(get_data_iterator_from(code_text), nn, device, parallelize=False)
    return list(predictions)


st.title("Model Explorer")
code_text = st.text_area(label="Code Text")
if len(code_text) > 0:
    predictions = predict(code_text)
    st.components.v1.html(predictions_to_html(predictions, include_header=False), height=400, width=900, scrolling=True)

    st.title("Graphs")
    for original_data, _, _ in predictions:
        ps = gv.Digraph(name="graph", node_attr={"shape": "rectangular"})

        options = st.multiselect(
            "Visualized Edges", list(original_data["graph"]["edges"]), default=list(original_data["graph"]["edges"])
        )

        nodes_with_edges = set()
        for edge_type, edges in original_data["graph"]["edges"].items():
            if edge_type not in options:
                continue
            for edge in edges:
                nodes_with_edges.add(edge[0])
                nodes_with_edges.add(edge[1])

        for i, node_name in enumerate(original_data["graph"]["nodes"]):
            if i not in nodes_with_edges:
                continue
            ps.node(f"n{i}", gv.escape(node_name))
        for edge_type, edges in original_data["graph"]["edges"].items():
            if edge_type not in options:
                continue
            for edge in edges:
                ps.edge(f"n{edge[0]}", f"n{edge[1]}", color=EDGE_COLORS.get(edge_type, "black"), label=edge_type)
        st.write(ps)
        st.write(original_data)
