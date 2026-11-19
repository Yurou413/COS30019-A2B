import pandas as pd

import app


def test_graph_for_single_site_has_no_neighbors():
    sites = pd.DataFrame(
        [{"scats_id": 1, "lat": -37.8, "lon": 145.0}]
    )

    assert app.build_scats_graph(sites) == {1: []}
