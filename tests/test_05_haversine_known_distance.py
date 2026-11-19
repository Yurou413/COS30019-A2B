import app


def test_haversine_distance_for_one_degree_of_longitude():
    distance = app.haversine_km(0.0, 0.0, 0.0, 1.0)

    assert abs(distance - 111.2) < 0.2
