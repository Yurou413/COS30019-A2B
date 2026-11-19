import tempfile
from unittest.mock import patch

import pandas as pd

import app


def test_scats_loading_filters_invalid_coordinates_and_averages():
    raw = pd.DataFrame(
        {
            "SCATS Number": [2000, 2000, 2000],
            "Location": ["Test intersection", "Other label", "Invalid row"],
            "NB_LATITUDE": [-37.8, -37.9, 0.0],
            "NB_LONGITUDE": [145.0, 145.2, 0.0],
        }
    )
    with tempfile.NamedTemporaryFile(suffix=".xls") as file:
        with patch.object(app.pd, "read_excel", return_value=raw):
            sites, used_fallback = app.load_scats_sites(file.name)

    assert not used_fallback
    assert len(sites) == 1
    assert sites.iloc[0]["scats_id"] == 2000
    assert abs(sites.iloc[0]["lat"] - (-37.85)) < 1e-9
    assert abs(sites.iloc[0]["lon"] - 145.1) < 1e-9
