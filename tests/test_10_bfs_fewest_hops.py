import app


def test_bfs_returns_a_fewest_hop_path():
    graph = {
        1: [(2, 10.0), (3, 0.2)],
        2: [(1, 10.0), (4, 10.0)],
        3: [(1, 0.2), (5, 0.2)],
        4: [(2, 10.0), (5, 0.2)],
        5: [(3, 0.2), (4, 0.2)],
    }
    lookup = {
        node: {"lat": -37.8 - node * 0.001, "lon": 145.0}
        for node in graph
    }

    path, _ = app._run_search(graph, lookup, 1, 4, "BFS", {})

    assert path == [1, 2, 4]
