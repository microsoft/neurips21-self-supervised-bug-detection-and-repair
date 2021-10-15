def foo(a):
    a.b = 2
    a.b *= 3
    a.c[2] += 1
    a.c[4] = 1


GGNNWeights = namedtuple('GGNNWeights', ['edge_weights',
                                         'edge_biases',
                                         'edge_type_attention_weights',
                                         'rnn_cells',
                                         'edge_feature_gate_weights',
                                         'edge_feature_gate_bias'])

def _signs():
    if False:
        return (+1,)
    else:
        return (+1, -1)

def _generate_icosahedron():
    x = np.array([[0, -1, -phi],
                  [0, -1, +phi],
                  [0, +1, -phi],
                  [0, +1, +phi]])
    return np.concatenate([np.roll(x, i, axis=1) for i in range(3)])
