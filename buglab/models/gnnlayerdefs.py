from ptgnn.neuralmodels.gnn.messagepassing import GatedMessagePassingLayer, MlpMessagePassingLayer
from ptgnn.neuralmodels.gnn.messagepassing.residuallayers import ConcatResidualLayer


def create_mlp_mp_layers(hidden_state_size, dropout_rate, num_edges: int, features_dimension: int = 0):
    mlp_mp_constructor = lambda: MlpMessagePassingLayer(
        input_state_dimension=hidden_state_size,
        message_dimension=hidden_state_size,
        output_state_dimension=hidden_state_size,
        num_edge_types=num_edges,
        message_aggregation_function="max",
        dropout_rate=dropout_rate,
        features_dimension=features_dimension,
    )
    mlp_mp_after_res_constructor = lambda: MlpMessagePassingLayer(
        input_state_dimension=2 * hidden_state_size,
        message_dimension=2 * hidden_state_size,
        output_state_dimension=hidden_state_size,
        num_edge_types=num_edges,
        message_aggregation_function="max",
        dropout_rate=dropout_rate,
        features_dimension=features_dimension,
    )
    r1 = ConcatResidualLayer(hidden_state_size)
    r2 = ConcatResidualLayer(hidden_state_size)
    return [
        r1.pass_through_dummy_layer(),
        mlp_mp_constructor(),
        mlp_mp_constructor(),
        mlp_mp_constructor(),
        r1,
        mlp_mp_after_res_constructor(),
        r2.pass_through_dummy_layer(),
        mlp_mp_constructor(),
        mlp_mp_constructor(),
        mlp_mp_constructor(),
        r2,
        mlp_mp_after_res_constructor(),
    ]


def create_ggnn_mp_layers(hidden_state_size, dropout_rate, num_edges: int):
    ggnn_mp = GatedMessagePassingLayer(
        state_dimension=hidden_state_size,
        message_dimension=hidden_state_size,
        num_edge_types=num_edges,
        message_aggregation_function="max",
        dropout_rate=dropout_rate,
    )
    r1 = ConcatResidualLayer(hidden_state_size)
    return [
        r1.pass_through_dummy_layer(),
        ggnn_mp,
        ggnn_mp,
        ggnn_mp,
        ggnn_mp,
        ggnn_mp,
        ggnn_mp,
        ggnn_mp,
        r1,
        GatedMessagePassingLayer(
            state_dimension=2 * hidden_state_size,
            message_dimension=hidden_state_size,
            num_edge_types=num_edges,
            message_aggregation_function="max",
            dropout_rate=dropout_rate,
        ),
    ]
