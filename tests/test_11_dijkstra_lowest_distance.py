import app


def test_dijkstra_chooses_lowest_total_distance():
    graph = {
        1: [(2, 2.0), (3, 0.5)],
        2: [(1, 2.0), (4, 0.5)],
        3: [(1, 0.5), (4, 5.0)],
        4: [(2, 0.5), (3, 5.0)],
    }
    lookup = {
        node: {"lat": -37.8 - node * 0.001, "lon": 145.0}
        for node in graph
    }

    path, _ = app._run_search(graph, lookup, 1, 4, "Dijkstra", {})

    assert path == [1, 2, 4]
