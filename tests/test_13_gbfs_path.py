import app


def test_gbfs_returns_a_path_to_destination():
    graph = {
        1: [(2, 100.0), (3, 1.0)],
        2: [(1, 100.0), (4, 100.0)],
        3: [(1, 1.0), (4, 1.0)],
        4: [(2, 100.0), (3, 1.0)],
    }
    lookup = {
        1: {"lat": 0.0, "lon": 0.00},
        2: {"lat": 0.0, "lon": 0.02},
        3: {"lat": 0.0, "lon": 0.005},
        4: {"lat": 0.0, "lon": 0.03},
    }

    path, _ = app._run_search(graph, lookup, 1, 4, "GBFS", {})

    assert path == [1, 2, 4]
