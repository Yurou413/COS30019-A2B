import json
import math
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
INPUT_FILE = DATA_DIR / "Scats Data October 2006.xls"
TIMESERIES_FILE = DATA_DIR / "traffic_timeseries.csv"
COMBINED_FILE = DATA_DIR / "combined_traffic_timeseries.csv"
GRAPH_FILE = DATA_DIR / "scats_graph.json"

TRAFFIC_COLUMNS = [f"V{i:02d}" for i in range(96)]
LOCATION_COLUMNS = [
    "SCATS Number",
    "Location",
    "NB_LATITUDE",
    "NB_LONGITUDE",
]


def load_and_reshape():
    data = pd.read_excel(INPUT_FILE, sheet_name="Data", header=1)

    required_columns = LOCATION_COLUMNS + ["Date"] + TRAFFIC_COLUMNS
    missing_columns = sorted(set(required_columns) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing columns in source workbook: {missing_columns}")

    site_ids = pd.to_numeric(data["SCATS Number"], errors="coerce")
    invalid_site_ids = site_ids.isna() | (site_ids % 1 != 0)
    if invalid_site_ids.any():
        raise ValueError(
            f"Found {int(invalid_site_ids.sum())} rows with invalid SCATS numbers"
        )
    data["SCATS Number"] = site_ids.astype("int64")

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    invalid_dates = data["Date"].isna()
    if invalid_dates.any():
        raise ValueError(
            f"Found {int(invalid_dates.sum())} rows with invalid dates"
        )

    data[TRAFFIC_COLUMNS] = data[TRAFFIC_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )

    long_data = data.melt(
        id_vars=LOCATION_COLUMNS + ["Date"],
        value_vars=TRAFFIC_COLUMNS,
        var_name="Interval",
        value_name="Traffic Volume",
    )
    long_data["Date"] = long_data["Date"].dt.normalize()
    interval_number = long_data["Interval"].str[1:].astype("int64")
    long_data["Timestamp"] = long_data["Date"] + pd.to_timedelta(
        interval_number * 15, unit="minutes"
    )
    long_data = long_data.sort_values(
        ["SCATS Number", "Timestamp"]
    ).reset_index(drop=True)

    combined_data = (
        long_data.groupby(
            ["SCATS Number", "Timestamp"], as_index=False
        )["Traffic Volume"]
        .sum(min_count=1)
        .sort_values(["SCATS Number", "Timestamp"])
        .reset_index(drop=True)
    )

    return data, long_data, combined_data


def haversine_m(point_a, point_b):
    lat1, lon1 = map(math.radians, point_a)
    lat2, lon2 = map(math.radians, point_b)
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    haversine = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )
    return 2 * 6_371_000 * math.asin(math.sqrt(min(1.0, haversine)))


def build_graph(location_data, neighbors_per_site=3):
    if neighbors_per_site < 1:
        raise ValueError("neighbors_per_site must be at least 1")

    locations = location_data[
        ["SCATS Number", "NB_LATITUDE", "NB_LONGITUDE"]
    ].copy()
    locations["NB_LATITUDE"] = pd.to_numeric(
        locations["NB_LATITUDE"], errors="coerce"
    )
    locations["NB_LONGITUDE"] = pd.to_numeric(
        locations["NB_LONGITUDE"], errors="coerce"
    )
    locations = locations.dropna(
        subset=["SCATS Number", "NB_LATITUDE", "NB_LONGITUDE"]
    )

    coordinate_counts = locations.groupby("SCATS Number")[
        ["NB_LATITUDE", "NB_LONGITUDE"]
    ].nunique()
    conflicting_sites = coordinate_counts[
        (coordinate_counts > 1).any(axis=1)
    ].index.tolist()
    if conflicting_sites:
        raise ValueError(
            "Sites have conflicting coordinates: "
            f"{conflicting_sites[:10]}"
        )

    locations = locations.drop_duplicates("SCATS Number")
    nodes = {
        int(row["SCATS Number"]): (
            float(row["NB_LATITUDE"]),
            float(row["NB_LONGITUDE"]),
        )
        for _, row in locations.iterrows()
    }
    site_ids = sorted(nodes)
    if len(site_ids) < 2:
        raise ValueError("At least two sites with coordinates are required")

    k = min(neighbors_per_site, len(site_ids) - 1)
    edge_costs = {site_id: {} for site_id in site_ids}
    for site_id in site_ids:
        nearest = sorted(
            (
                (other_id, haversine_m(nodes[site_id], nodes[other_id]))
                for other_id in site_ids
                if other_id != site_id
            ),
            key=lambda item: item[1],
        )[:k]

        for other_id, distance in nearest:
            edge_costs[site_id][other_id] = distance
            edge_costs[other_id][site_id] = distance

    graph = {
        site_id: sorted(neighbors.items())
        for site_id, neighbors in edge_costs.items()
    }
    return nodes, graph


def save_graph(nodes, graph):
    payload = {
        "description": (
            "Undirected nearest-neighbor approximation based on site "
            "coordinates; edges are not verified road connections."
        ),
        "edge_cost_unit": "meters",
        "nodes": {
            str(site_id): {
                "latitude": coordinates[0],
                "longitude": coordinates[1],
            }
            for site_id, coordinates in nodes.items()
        },
        "graph": {
            str(site_id): [
                [neighbor_id, distance]
                for neighbor_id, distance in neighbors
            ]
            for site_id, neighbors in graph.items()
        },
    }
    with GRAPH_FILE.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)


def load_graph(path=GRAPH_FILE):
    with Path(path).open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    nodes = {
        int(site_id): (
            float(coordinates["latitude"]),
            float(coordinates["longitude"]),
        )
        for site_id, coordinates in payload["nodes"].items()
    }
    graph = {
        int(site_id): [
            (int(neighbor_id), float(distance))
            for neighbor_id, distance in neighbors
        ]
        for site_id, neighbors in payload["graph"].items()
    }
    return nodes, graph


def main():
    source_data, long_data, combined_data = load_and_reshape()

    long_data.to_csv(TIMESERIES_FILE, index=False)
    combined_data.to_csv(COMBINED_FILE, index=False)

    nodes, graph = build_graph(source_data, neighbors_per_site=3)
    save_graph(nodes, graph)

    edge_count = sum(len(neighbors) for neighbors in graph.values()) // 2
    print(f"Traffic time series saved to: {TIMESERIES_FILE}")
    print(f"Combined time series saved to: {COMBINED_FILE}")
    print(f"Graph saved to: {GRAPH_FILE}")
    print(f"Traffic rows: {len(long_data):,}")
    print(f"Combined site/timestamp rows: {len(combined_data):,}")
    print(f"Graph nodes: {len(nodes):,}; undirected edges: {edge_count:,}")
    print("Graph edge costs are geographic distances in meters.")


if __name__ == "__main__":
    main()
