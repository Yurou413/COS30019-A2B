import app


def test_load_config_reads_default_settings():
    config = app.load_config()

    assert config["default_model"] == "LSTM"
    assert "A*" in config["available_algorithms"]
