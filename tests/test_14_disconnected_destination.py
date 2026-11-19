import app


def test_search_returns_none_when_destination_is_disconnected():
    graph = {1: [(2, 1.0)], 2: [(1, 1.0)], 3: []}
    lookup = {
        node: {"lat": -37.8 - node * 0.001, "lon": 145.0}
        for node in graph
    }

    path, nodes_created = app._run_search(
        graph, lookup, 1, 3, "Dijkstra", {}
    )

    assert path is None
    assert nodes_created >= 1
