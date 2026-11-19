import app


def test_haversine_distance_is_zero_for_identical_points():
    assert app.haversine_km(-37.8, 145.0, -37.8, 145.0) == 0.0
