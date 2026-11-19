import pandas as pd

import app


def test_graph_edges_are_symmetric():
    sites = pd.DataFrame(
        [
            {"scats_id": 1, "lat": -37.8, "lon": 145.0},
            {"scats_id": 2, "lat": -37.81, "lon": 145.01},
        ]
    )
    graph = app.build_scats_graph(sites, k_neighbors=1)
    forward = dict(graph[1])[2]
    reverse = dict(graph[2])[1]

    assert abs(forward - reverse) < 1e-9
