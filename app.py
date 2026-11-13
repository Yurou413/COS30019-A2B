"""
COS30019 Introduction to AI - Assignment 2 Part B
Traffic-Based Route Guidance System (TBRGS) for Boroondara, Melbourne.

Streamlit + Folium GUI with:
- Origin / Destination SCATS site selection
- Date & time selection
- ML model selection (LSTM, GRU, Regression Tree) + training trigger
- Configurable traffic parameters (speed limit, intersection delay, max routes)
- Interactive OpenStreetMap (Folium) with VicRoads lat/lon calibration sliders
- Modular `get_routes()` hook ready for backend integration
"""

from __future__ import annotations

from collections import deque
import datetime as dt_mod
import heapq
import json
import math
from pathlib import Path
from typing import Any

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

CONFIG_PATH = Path(__file__).parent / "config.json"

FALLBACK_SITES = [
    {"scats_id": 2000, "location": "WARRIGAL_RD N of TOORAK_RD", "lat": -37.851923, "lon": 145.094324},
    {"scats_id": 3001, "location": "HIGH_ST NE of BARKERS_RD", "lat": -37.814565, "lon": 145.022210},
    {"scats_id": 3002, "location": "DENMARK_ST N of BARKERS_RD", "lat": -37.815162, "lon": 145.026572},
    {"scats_id": 3120, "location": "BURKE_RD N of CANTERBURY_RD", "lat": -37.822895, "lon": 145.057288},
    {"scats_id": 3127, "location": "BALWYN_RD N of CANTERBURY_RD", "lat": -37.825060, "lon": 145.078000},
]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load default GUI and routing settings from config.json."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading Boroondara SCATS sites...")
def load_scats_sites(data_path: str, sheet_name: str = "Data") -> tuple[pd.DataFrame, bool]:
    """
    Load unique Boroondara SCATS sites and coordinates from the VicRoads dataset.

    Filters out invalid (0.0, 0.0) coordinates (e.g., site 4266 has one zeroed row)
    before averaging approach coordinates per SCATS Number.
    Returns (sites_df, used_fallback).
    """
    file_path = Path(__file__).parent / data_path
    if not file_path.exists():
        return pd.DataFrame(FALLBACK_SITES), True

    try:
        raw_df = pd.read_excel(file_path, sheet_name=sheet_name, header=1)
        valid_coords = raw_df[
            (raw_df["NB_LATITUDE"].notna())
            & (raw_df["NB_LONGITUDE"].notna())
            & (raw_df["NB_LATITUDE"] < -37.0)
            & (raw_df["NB_LONGITUDE"] > 144.0)
        ]

        sites_df = (
            valid_coords.groupby("SCATS Number", as_index=False)
            .agg(
                location=("Location", "first"),
                lat=("NB_LATITUDE", "mean"),
                lon=("NB_LONGITUDE", "mean"),
            )
            .rename(columns={"SCATS Number": "scats_id"})
            .sort_values("scats_id")
            .reset_index(drop=True)
        )
        sites_df["scats_id"] = sites_df["scats_id"].astype(int)
        return sites_df, False
    except Exception:
        return pd.DataFrame(FALLBACK_SITES), True


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance in kilometres between two lat/lon points."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def flow_to_speed_kmh(flow_veh_hr: float, speed_limit_kmh: float = 60.0) -> float:
    """
    Convert traffic flow (veh/hr) to speed (km/h) per VicRoads formula:
    - flow <= 351 veh/hr: capped at speed_limit_kmh (default 60 km/h)
    - flow > 351 veh/hr: under-capacity (green) branch of
      flow = -1.4648375 * speed^2 + 93.75 * speed
    """
    if flow_veh_hr <= 351.0:
        return speed_limit_kmh

    a = -1.4648375
    b = 93.75
    c = -flow_veh_hr
    discriminant = max(0.0, b * b - 4 * a * c)
    # Under-capacity (upper/green) branch uses the larger speed root
    speed = (-b - math.sqrt(discriminant)) / (2 * a)
    return max(5.0, min(speed_limit_kmh, speed))


def train_selected_model(model_name: str) -> str:
    """
    Placeholder hook to trigger ML model training from the GUI.
    Swap with Yurou (LSTM), Huiming (GRU), and Vivian (Regression Tree) training calls.
    """
    return (
        f"Training hook triggered for **{model_name}**. "
        "Connect `ML-models/` training functions here once team models are finalized."
    )


def build_scats_graph(
    sites_df: pd.DataFrame,
    k_neighbors: int = 4,
) -> dict[int, list[tuple[int, float]]]:
    """
    Build an undirected SCATS intersection graph by connecting each site
    to its `k_neighbors` nearest SCATS intersections (road-network proxy).
    """
    rows = sites_df.to_dict(orient="records")
    graph: dict[int, dict[int, float]] = {int(r["scats_id"]): {} for r in rows}

    for r1 in rows:
        id1 = int(r1["scats_id"])
        dists = []
        for r2 in rows:
            id2 = int(r2["scats_id"])
            if id1 == id2:
                continue
            d = haversine_km(r1["lat"], r1["lon"], r2["lat"], r2["lon"])
            dists.append((d, id2))
        dists.sort(key=lambda x: x[0])
        for d, id2 in dists[:k_neighbors]:
            graph[id1][id2] = d
            graph[id2][id1] = d

    return {node: list(nbrs.items()) for node, nbrs in graph.items()}


def _run_search(
    graph: dict[int, list[tuple[int, float]]],
    site_lookup: dict[int, dict[str, Any]],
    origin: int,
    destination: int,
    algorithm_name: str,
    penalties: dict[tuple[int, int], float],
) -> tuple[list[int] | None, int]:
    """
    Run a single pathfinding search from origin to destination using the
    selected algorithm ('A*', 'Dijkstra', 'GBFS', 'BFS', or 'DFS').
    Returns (path, nodes_created).
    """
    dest_lat = site_lookup[destination]["lat"]
    dest_lon = site_lookup[destination]["lon"]

    def heuristic_km(node_id: int) -> float:
        return haversine_km(
            site_lookup[node_id]["lat"],
            site_lookup[node_id]["lon"],
            dest_lat,
            dest_lon,
        )

    if algorithm_name == "BFS":
        queue = deque([(origin, [origin])])
        visited = {origin}
        nodes_created = 1
        while queue:
            node, path = queue.popleft()
            if node == destination:
                return path, nodes_created
            neighbors = sorted(
                graph.get(node, []),
                key=lambda e: (penalties.get((min(node, e[0]), max(node, e[0])), 1.0), e[0]),
            )
            for nbr, _ in neighbors:
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, path + [nbr]))
                    nodes_created += 1
        return None, nodes_created

    if algorithm_name == "DFS":
        stack: list[tuple[int, list[int]]] = [(origin, [origin])]
        nodes_created = 1
        while stack:
            node, path = stack.pop()
            if node == destination:
                return path, nodes_created
            neighbors = sorted(
                graph.get(node, []),
                key=lambda e: (penalties.get((min(node, e[0]), max(node, e[0])), 1.0), e[0]),
                reverse=True,
            )
            for nbr, _ in neighbors:
                if nbr not in path:
                    stack.append((nbr, path + [nbr]))
                    nodes_created += 1
                    if nodes_created > 5000:
                        return None, nodes_created
        return None, nodes_created

    # Priority-queue algorithms: A*, Dijkstra, GBFS
    seq = 0
    start_h = heuristic_km(origin) if algorithm_name in ("A*", "GBFS") else 0.0
    pq: list[tuple[float, int, float, int, list[int]]] = [(start_h, seq, 0.0, origin, [origin])]
    best_g: dict[int, float] = {origin: 0.0}
    visited_gbfs: set[int] = set()
    nodes_created = 1

    while pq:
        _, _, g_cost, node, path = heapq.heappop(pq)
        if node == destination:
            return path, nodes_created

        if algorithm_name == "GBFS":
            if node in visited_gbfs:
                continue
            visited_gbfs.add(node)
        elif g_cost > best_g.get(node, float("inf")):
            continue

        for nbr, dist_km in sorted(graph.get(node, []), key=lambda e: e[0]):
            if nbr in path:
                continue
            edge = (min(node, nbr), max(node, nbr))
            penalty = penalties.get(edge, 1.0)
            new_g = g_cost + dist_km * penalty

            if algorithm_name == "GBFS":
                if nbr in visited_gbfs:
                    continue
                priority = heuristic_km(nbr) * penalty
            elif algorithm_name == "A*":
                if new_g >= best_g.get(nbr, float("inf")):
                    continue
                best_g[nbr] = new_g
                priority = new_g + heuristic_km(nbr)
            else:  # Dijkstra
                if new_g >= best_g.get(nbr, float("inf")):
                    continue
                best_g[nbr] = new_g
                priority = new_g

            seq += 1
            heapq.heappush(pq, (priority, seq, new_g, nbr, path + [nbr]))
            nodes_created += 1

    return None, nodes_created


def find_top_k_paths(
    graph: dict[int, list[tuple[int, float]]],
    site_lookup: dict[int, dict[str, Any]],
    origin: int,
    destination: int,
    k: int,
    algorithm_name: str = "A*",
) -> list[tuple[list[int], int]]:
    """
    Find up to `k` distinct multi-hop SCATS paths using the chosen search
    algorithm and iterative edge penalization. Returns list of (path, nodes_created).
    """
    results: list[tuple[list[int], int]] = []
    seen: set[tuple[int, ...]] = set()
    penalties: dict[tuple[int, int], float] = {}

    for _ in range(k * 5):
        path, nodes_created = _run_search(
            graph, site_lookup, origin, destination, algorithm_name, penalties
        )
        if not path:
            break
        key = tuple(path)
        if key not in seen:
            seen.add(key)
            results.append((path, nodes_created))
            if len(results) >= k:
                break
        for i in range(len(path) - 1):
            edge = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
            penalties[edge] = penalties.get(edge, 1.0) * 1.65

    return results


def get_routes(
    origin_id: int,
    destination_id: int,
    travel_datetime: dt_mod.datetime,
    model_name: str,
    algorithm_name: str,
    speed_limit_kmh: float,
    intersection_delay_sec: float,
    max_routes: int,
    sites_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Modular route-finding & travel-time estimation interface.

    TODO (End-to-End Integration):
        Replace this placeholder body with Vivian's ported Assignment 2A
        k-shortest pathfinding + ML flow prediction backend.

    Returns a list of up to `max_routes` dicts sorted by travel_time_sec:
        {
            "rank": int,
            "path": list[int],
            "distance_km": float,
            "travel_time_sec": float,
            "intersections": int,
            "nodes_created": int,
            "algorithm": str,
            "model": str,
        }
    """
    site_lookup = sites_df.set_index("scats_id").to_dict(orient="index")
    graph = build_scats_graph(sites_df, k_neighbors=4)
    candidate_results = find_top_k_paths(
        graph, site_lookup, origin_id, destination_id, max_routes, algorithm_name
    )

    # Slight deterministic flow variation by hour & model for realistic UI preview
    hour_factor = 1.0 + 0.35 * math.sin((travel_datetime.hour - 6) / 24.0 * 2 * math.pi)
    model_bias = {"LSTM": 1.0, "GRU": 1.03, "Regression Tree": 0.97}.get(model_name, 1.0)
    base_flow = 320.0 * hour_factor * model_bias

    routes = []
    for idx, (path, nodes_created) in enumerate(candidate_results):
        total_dist_km = sum(
            haversine_km(
                site_lookup[path[i]]["lat"],
                site_lookup[path[i]]["lon"],
                site_lookup[path[i + 1]]["lat"],
                site_lookup[path[i + 1]]["lon"],
            )
            for i in range(len(path) - 1)
        )
        # Apply a 1.15x road-network winding factor for placeholder realism
        road_dist_km = total_dist_km * 1.15
        est_flow = base_flow + idx * 45.0
        speed_kmh = flow_to_speed_kmh(est_flow, speed_limit_kmh)
        intersections = max(0, len(path) - 1)
        travel_time_sec = (road_dist_km / speed_kmh) * 3600.0 + (
            intersections * intersection_delay_sec
        )

        routes.append(
            {
                "rank": idx + 1,
                "path": path,
                "distance_km": round(road_dist_km, 2),
                "travel_time_sec": round(travel_time_sec, 1),
                "intersections": intersections,
                "nodes_created": nodes_created,
                "algorithm": algorithm_name,
                "model": model_name,
            }
        )

    routes.sort(key=lambda r: r["travel_time_sec"])
    for rank, route in enumerate(routes, start=1):
        route["rank"] = rank
    return routes


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_road_geometry(waypoints: tuple[tuple[float, float], ...]) -> list[tuple[float, float]]:
    """
    Snap SCATS intersection waypoints to real OpenStreetMap roads via OSRM.
    Falls back to the straight waypoint polyline if the API is unreachable.
    """
    if len(waypoints) < 2:
        return list(waypoints)

    coords_str = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in waypoints)
    url = (
        f"https://router.project-osrm.org/route/v1/driving/{coords_str}"
        "?overview=full&geometries=geojson"
    )
    try:
        resp = requests.get(url, timeout=4)
        resp.raise_for_status()
        data = resp.json()
        coords = data["routes"][0]["geometry"]["coordinates"]
        return [(lat, lon) for lon, lat in coords]
    except Exception:
        return list(waypoints)


def format_duration(seconds: float) -> str:
    """Format duration in seconds as 'X min YY s'."""
    mins, secs = divmod(int(round(seconds)), 60)
    return f"{mins} min {secs:02d} s"


def build_map(
    sites_df: pd.DataFrame,
    origin_id: int,
    destination_id: int,
    selected_route: dict[str, Any] | None,
    center_lat: float,
    center_lon: float,
    zoom: int,
    lat_offset: float,
    lon_offset: float,
) -> folium.Map:
    """Build an interactive Folium OpenStreetMap with calibrated SCATS sites and route."""
    fmap = folium.Map(
        location=[center_lat + lat_offset, center_lon + lon_offset],
        zoom_start=zoom,
        tiles="OpenStreetMap",
    )

    selected_path_set = set(selected_route["path"]) if selected_route else set()
    calibrated_coords: dict[int, tuple[float, float]] = {}
    for _, row in sites_df.iterrows():
        sid = int(row["scats_id"])
        lat = float(row["lat"]) + lat_offset
        lon = float(row["lon"]) + lon_offset
        calibrated_coords[sid] = (lat, lon)

        if sid == origin_id:
            color, radius, prefix = "green", 9, "Origin"
        elif sid == destination_id:
            color, radius, prefix = "red", 9, "Destination"
        elif sid in selected_path_set:
            color, radius, prefix = "#e65100", 7, "Via SCATS"
        else:
            color, radius, prefix = "#3388ff", 5, "SCATS"

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=f"<b>{prefix} {sid}</b><br>{row['location']}<br>({lat:.5f}, {lon:.5f})",
            tooltip=f"{sid} — {row['location']}",
        ).add_to(fmap)

    if selected_route and len(selected_route["path"]) >= 2:
        route_waypoints = tuple(
            calibrated_coords[sid]
            for sid in selected_route["path"]
            if sid in calibrated_coords
        )
        if len(route_waypoints) >= 2:
            road_points = fetch_road_geometry(route_waypoints)
            folium.PolyLine(
                locations=road_points,
                color="#e65100",
                weight=5,
                opacity=0.85,
                tooltip=(
                    f"Route #{selected_route['rank']} | "
                    f"{format_duration(selected_route['travel_time_sec'])}"
                ),
            ).add_to(fmap)

    return fmap


def main() -> None:
    st.set_page_config(
        page_title="Boroondara TBRGS — COS30019",
        page_icon="🚦",
        layout="wide",
    )

    config = load_config()
    sites_df, used_fallback = load_scats_sites(
        config["scats_data_path"],
        config.get("scats_sheet_name", "Data"),
    )

    st.title("🚦 Traffic-Based Route Guidance System (TBRGS)")
    st.caption("Boroondara, Melbourne — COS30019 Introduction to AI (Assignment 2 Part B)")

    if used_fallback:
        st.warning(
            "Could not load `data/Scats Data October 2006.xls`; displaying fallback SCATS sites."
        )

    site_ids = sites_df["scats_id"].tolist()
    site_labels = {
        int(row["scats_id"]): f"{int(row['scats_id'])} — {row['location']}"
        for _, row in sites_df.iterrows()
    }

    default_origin = (
        config["default_origin"] if config["default_origin"] in site_ids else site_ids[0]
    )
    default_dest = (
        config["default_destination"]
        if config["default_destination"] in site_ids
        else site_ids[-1]
    )

    with st.sidebar:
        st.header("Route & Model Controls")

        origin_id = st.selectbox(
            "Origin SCATS Site (O)",
            options=site_ids,
            index=site_ids.index(default_origin),
            format_func=lambda sid: site_labels[sid],
        )
        destination_id = st.selectbox(
            "Destination SCATS Site (D)",
            options=site_ids,
            index=site_ids.index(default_dest),
            format_func=lambda sid: site_labels[sid],
        )

        st.subheader("Departure Date & Time")
        travel_date = st.date_input(
            "Date",
            value=dt_mod.date(2006, 10, 1),
        )
        travel_time = st.time_input(
            "Time (15-min SCATS interval)",
            value=dt_mod.time(8, 0),
            step=dt_mod.timedelta(minutes=15),
        )
        travel_dt = dt_mod.datetime.combine(travel_date, travel_time)

        st.subheader("ML Prediction Model")
        models = config.get("available_models", ["LSTM", "GRU", "Regression Tree"])
        default_model = config.get("default_model", models[0])
        model_name = st.selectbox(
            "Model",
            options=models,
            index=models.index(default_model) if default_model in models else 0,
        )
        if st.button("Train Selected Model", use_container_width=True):
            st.info(train_selected_model(model_name))

        st.subheader("Search Algorithm")
        algorithms = config.get(
            "available_algorithms", ["A*", "Dijkstra", "GBFS", "BFS", "DFS"]
        )
        default_algo = config.get("default_algorithm", algorithms[0])
        algorithm_name = st.selectbox(
            "Pathfinding Algorithm",
            options=algorithms,
            index=algorithms.index(default_algo) if default_algo in algorithms else 0,
        )

        st.subheader("Traffic Parameters")
        speed_limit_kmh = st.number_input(
            "Speed Limit (km/h)",
            min_value=20,
            max_value=110,
            value=int(config.get("speed_limit_kmh", 60)),
            step=5,
        )
        intersection_delay_sec = st.number_input(
            "Intersection Delay (seconds)",
            min_value=0,
            max_value=120,
            value=int(config.get("intersection_delay_sec", 30)),
            step=5,
        )
        max_routes = st.slider(
            "Top N Routes",
            min_value=1,
            max_value=5,
            value=int(config.get("max_routes", 5)),
        )

        st.subheader("Map Calibration (Research Bonus)")
        if "lat_offset" not in st.session_state:
            st.session_state["lat_offset"] = float(config.get("lat_offset", 0.0))
        if "lon_offset" not in st.session_state:
            st.session_state["lon_offset"] = float(config.get("lon_offset", 0.0))

        lat_offset = st.slider(
            "Latitude Offset",
            min_value=-0.0100,
            max_value=0.0100,
            step=0.0001,
            format="%.4f",
            key="lat_offset",
        )
        lon_offset = st.slider(
            "Longitude Offset",
            min_value=-0.0100,
            max_value=0.0100,
            step=0.0001,
            format="%.4f",
            key="lon_offset",
        )
        if st.button("Reset Map Calibration", use_container_width=True):
            st.session_state["lat_offset"] = 0.0
            st.session_state["lon_offset"] = 0.0
            st.rerun()

    if origin_id == destination_id:
        st.error("Origin and Destination SCATS sites must be different.")
        routes: list[dict[str, Any]] = []
    else:
        routes = get_routes(
            origin_id=origin_id,
            destination_id=destination_id,
            travel_datetime=travel_dt,
            model_name=model_name,
            algorithm_name=algorithm_name,
            speed_limit_kmh=float(speed_limit_kmh),
            intersection_delay_sec=float(intersection_delay_sec),
            max_routes=int(max_routes),
            sites_df=sites_df,
        )

    col_map, col_routes = st.columns([3, 2], gap="large")

    with col_routes:
        st.subheader(f"Top {len(routes)} Fastest Routes ({algorithm_name} + {model_name})")
        if routes:
            selected_rank = st.radio(
                "Highlight Route on Map",
                options=[r["rank"] for r in routes],
                format_func=lambda rk: (
                    f"Route #{rk} — {format_duration(routes[rk - 1]['travel_time_sec'])} "
                    f"({routes[rk - 1]['distance_km']} km)"
                ),
            )
            selected_route = routes[selected_rank - 1]

            table_rows = [
                {
                    "Rank": r["rank"],
                    "SCATS Path": " → ".join(str(s) for s in r["path"]),
                    "Distance (km)": r["distance_km"],
                    "Est. Time": format_duration(r["travel_time_sec"]),
                    "Intersections": r["intersections"],
                    "Nodes": r["nodes_created"],
                    "Algorithm": r["algorithm"],
                    "Model": r["model"],
                }
                for r in routes
            ]
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

            st.caption(
                "⚠️ Currently using placeholder `get_routes()` math. "
                "Ready to swap in Vivian's A2A pathfinding + ML predictions."
            )
        else:
            selected_route = None
            st.info("Select distinct Origin and Destination sites to view routes.")

    with col_map:
        st.subheader("Boroondara SCATS Network Map")
        fmap = build_map(
            sites_df=sites_df,
            origin_id=origin_id,
            destination_id=destination_id,
            selected_route=selected_route,
            center_lat=float(config["map_center"]["lat"]),
            center_lon=float(config["map_center"]["lon"]),
            zoom=int(config.get("default_zoom", 13)),
            lat_offset=float(lat_offset),
            lon_offset=float(lon_offset),
        )
        st_folium(fmap, width=None, height=560, returned_objects=[])


if __name__ == "__main__":
    main()

