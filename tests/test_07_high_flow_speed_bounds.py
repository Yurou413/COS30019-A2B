import pytest

import app


def test_high_flow_speed_matches_vicroads_under_capacity_solution():
    speed = app.flow_to_speed_kmh(800.0, 60.0)

    assert speed == pytest.approx(53.86, abs=0.02)
