import app


def test_low_flow_uses_speed_limit():
    assert app.flow_to_speed_kmh(351.0, 50.0) == 50.0
