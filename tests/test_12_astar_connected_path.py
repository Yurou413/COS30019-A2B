import app


def test_astar_returns_a_connected_path_to_destination():
    graph = {
        1: [(2, 1.2), (3, 3.7)],
        2: [(1, 1.2), (4, 3.5)],
        3: [(1, 3.7), (4, 1.2)],
        4: [(2, 3.5), (3, 1.2)],
    }
    lookup = {
        1: {"lat": 0.0, "lon": 0.00},
        2: {"lat": 0.0, "lon": 0.01},
        3: {"lat": 0.0, "lon": 0.03},
        4: {"lat": 0.0, "lon": 0.04},
    }

    path, _ = app._run_search(graph, lookup, 1, 4, "A*", {})

    assert path == [1, 2, 4]
    assert all(b in dict(graph[a]) for a, b in zip(path, path[1:]))
