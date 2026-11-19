import datetime as dt
import math

import pandas as pd

import app


def test_top_k_finds_distinct_paths_and_route_time_matches_inputs():
    graph = {
        1: [(2, 1.0), (3, 1.0)],
        2: [(1, 1.0), (4, 1.0)],
        3: [(1, 1.0), (4, 1.0)],
        4: [(2, 1.0), (3, 1.0)],
    }
    lookup = {
        1: {"lat": -37.80, "lon": 145.00},
        2: {"lat": -37.81, "lon": 145.00},
        3: {"lat": -37.80, "lon": 145.01},
        4: {"lat": -37.81, "lon": 145.01},
    }
    paths = app.find_top_k_paths(graph, lookup, 1, 4, 2, "Dijkstra")
    sites = pd.DataFrame(
        [
            {"scats_id": node, **lookup[node], "location": f"Site {node}"}
            for node in (1, 4)
        ]
    )
    travel_datetime = dt.datetime(2006, 10, 1, 8)
    routes = app.get_routes(
        1,
        4,
        travel_datetime,
        "LSTM",
        "A*",
        60.0,
        30.0,
        1,
        sites,
    )

    assert [path for path, _ in paths] == [[1, 2, 4], [1, 3, 4]]
    assert len(routes) == 1
    route = routes[0]
    assert route["path"] == [1, 4]
    assert route["rank"] == 1
    assert route["intersections"] == 1
    assert route["algorithm"] == "A*"
    assert route["model"] == "LSTM"

    road_distance_km = (
        app.haversine_km(
            lookup[1]["lat"],
            lookup[1]["lon"],
            lookup[4]["lat"],
            lookup[4]["lon"],
        )
        * 1.15
    )
    estimated_flow = 320.0 * (
        1.0
        + 0.35
        * math.sin((travel_datetime.hour - 6) / 24.0 * 2 * math.pi)
    )
    speed_kmh = app.flow_to_speed_kmh(estimated_flow, 60.0)
    expected_time_sec = round(road_distance_km / speed_kmh * 3600.0 + 30.0, 1)

    assert route["distance_km"] == round(road_distance_km, 2)
    assert route["travel_time_sec"] == expected_time_sec
